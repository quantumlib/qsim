// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CPU backend of the gate-batched runner (gate_batch_runner.h), and the
// QSimGateBatchRunner entry point built on it.
//
// A tile is 2^tile_qubits contiguous amplitudes of the SIMD state
// space, sized to stay cache-resident. Each gate batch runs every tile
// through a sequential SIMD simulator, tiles in parallel over OpenMP
// threads, optionally splitting each tile across one SMT sibling team.
// Remaps use ApplyBitPairSwaps (qubit_remap.h), which moves whole lane
// groups, so the SIMD lane positions are pinned.

#ifndef RUN_QSIM_GATE_BATCH_H_
#define RUN_QSIM_GATE_BATCH_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu_thread_topology.h"
#include "gate_batch_runner.h"
#include "qubit_remap.h"
#include "util.h"

namespace qsim {

namespace gate_batch_internal {

// The only OpenMP-dependent code in this file. Without OpenMP the pragmas are
// ignored, so a parallel region runs once on the calling thread.
#ifdef _OPENMP
inline constexpr bool kHasOpenMP = true;
inline unsigned ParallelThreadCount() { return omp_get_num_threads(); }
inline unsigned ParallelThreadId() { return omp_get_thread_num(); }
#else
inline constexpr bool kHasOpenMP = false;
inline unsigned ParallelThreadCount() { return 1; }
inline unsigned ParallelThreadId() { return 0; }
#endif

// Geometry of the tiled state: an n-qubit state is viewed as
// 2^(n - L) contiguous tiles of 2^L amplitudes.
template <typename StateSpace>
struct TilePartition {
  TilePartition(unsigned state_qubits, unsigned requested_tile_qubits,
                unsigned num_threads, unsigned min_tile_qubits)
      : num_state_qubits(state_qubits),
        requested_tile_qubits(
            std::min(requested_tile_qubits, state_qubits)),
        tile_qubits(ChooseTileQubits(
            state_qubits, requested_tile_qubits, num_threads,
            min_tile_qubits)),
        floats_per_tile(StateSpace::MinSize(tile_qubits)),
        num_tiles(int64_t{1} << (state_qubits - tile_qubits)) {}

  // Use the requested L as an upper bound. For a small state, lower L until
  // there are at least bit_floor(num_threads) tiles for the outer parallel
  // loop. Using bit_floor rather than bit_ceil avoids doubling the number of
  // gate batches and remaps merely to occupy the last few threads.
  static unsigned ChooseTileQubits(unsigned state_qubits,
                                   unsigned requested_tile_qubits,
                                   unsigned num_threads,
                                   unsigned min_tile_qubits) {
    unsigned parallel_bits = 0;  // floor(log2(num_threads))
    while ((2u << parallel_bits) <= num_threads) ++parallel_bits;

    const auto parallel_tile_qubits =
        state_qubits > parallel_bits ? state_qubits - parallel_bits : 0u;
    const auto minimum = std::min(min_tile_qubits, state_qubits);
    return std::max(
        minimum,
        std::min({requested_tile_qubits, state_qubits,
                  parallel_tile_qubits}));
  }

  unsigned num_state_qubits;       // n
  unsigned requested_tile_qubits;  // User-specified upper bound for L.
  unsigned tile_qubits;            // Effective L.
  uint64_t floats_per_tile;        // State floats per tile (SIMD layout).
  int64_t num_tiles;               // 2^(n - L).
};

// Reusable synchronization for the SMT siblings working on one tile.
// Keep barriers on separate cache lines so independent cores never contend on
// the same coherence line.
struct alignas(64) SmtTeamBarrier {
  void Wait(unsigned team_size) {
    const auto current_generation = generation.load(std::memory_order_acquire);
    if (arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == team_size) {
      arrivals.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
    } else {
      // Spin: waits are microseconds; pause/yield measured 1-3% slower.
      while (generation.load(std::memory_order_acquire) ==
             current_generation) {
      }
    }
  }

  std::atomic<unsigned> arrivals{0};
  std::atomic<unsigned> generation{0};
};

// Where one OpenMP thread sits in the SMT team grid. Threads left over when
// the thread count is not divisible by the team size are inactive; normal SMT
// use is an exact 2-way split.
struct SmtTeamAssignment {
  unsigned team_size;
  unsigned num_teams;
  unsigned team_id;
  unsigned lane;
  bool active;
};

// Assigns OpenMP thread thread_id of num_threads to a team of
// inner_threads SMT siblings.
inline SmtTeamAssignment AssignSmtTeam(unsigned inner_threads,
                                       unsigned num_threads,
                                       unsigned thread_id) {
  const auto team_size = std::max(1u, std::min(inner_threads, num_threads));
  const auto num_teams = num_threads / team_size;
  const auto team_id = thread_id / team_size;
  return {team_size, num_teams, team_id, thread_id % team_size,
          team_id < num_teams};
}

// The For of the per-tile simulator. Each SMT team member runs its own
// slice of every kernel loop on the team's shared tile;
// ExecuteGatesOnTile sets the member's lane and synchronizes the team
// between gates. A team of one runs the whole loop.
struct CooperativeFor {
  explicit CooperativeFor(unsigned num_threads) { (void) num_threads; }

  static void Configure(unsigned team_size, unsigned team_thread_id) {
    team_size_ = team_size;
    team_thread_id_ = team_thread_id;
  }

  template <typename Function, typename... Args>
  static void Run(uint64_t size, Function&& func, Args&&... args) {
    const auto begin = size * team_thread_id_ / team_size_;
    const auto end = size * (team_thread_id_ + 1) / team_size_;
    for (uint64_t i = begin; i < end; ++i) {
      func(team_size_, team_thread_id_, i, args...);
    }
  }

 private:
  inline static thread_local unsigned team_size_ = 1;
  inline static thread_local unsigned team_thread_id_ = 0;
};

// Simulator with its For replaced by NewFor, e.g. SimulatorNEON<ParallelFor>
// becomes SimulatorNEON<CooperativeFor>. Extra template arguments such as
// SimulatorBasic's float type are kept.
template <typename Simulator, typename NewFor>
struct ReplaceFor;

template <template <typename...> class SimulatorT, typename For,
          typename NewFor, typename... Rest>
struct ReplaceFor<SimulatorT<For, Rest...>, NewFor> {
  using type = SimulatorT<NewFor, Rest...>;
};

}  // namespace gate_batch_internal

template <typename IO, typename Factory>
class CpuGateBatchBackend {
 public:
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  // Simulates one tile on the calling thread, or on a team of SMT
  // siblings that split each gate.
  using SeqSimulator = typename gate_batch_internal::ReplaceFor<
      typename Factory::Simulator, gate_batch_internal::CooperativeFor>::type;
  using SeqStateSpace = typename SeqSimulator::StateSpace;
  using ExecutableGate = gate_batch_internal::ExecutableGate<fp_type>;

  struct Parameter {
    unsigned num_threads = 1;
    unsigned inner_threads = 1;
    std::vector<unsigned> team_thread_cpus;
  };

  static constexpr unsigned kDefaultTileQubits = 19;

  // qsim's CPU simulators apply gates of up to six qubits.
  static constexpr unsigned kMaxGateQubits = 6;

  // Swap-pass spans are 2^(floor - lane_qubits) lane groups; 5 was the best
  // trade-off between remap budget and span length on the tested machines.
  static constexpr unsigned kDefaultEvictionFloor = 5;

  template <typename RunnerParameter>
  CpuGateBatchBackend(const RunnerParameter& param, unsigned num_qubits,
                      State& state)
      : param_(param),
        verbosity_(param.verbosity),
        partition_(num_qubits, param.tile_qubits, param.num_threads,
                   MinTileQubits(param.max_fused_size)),
        state_data_(state.get()),
        seq_sim_(1) {
    assert(partition_.tile_qubits >= std::min(kLaneQubits, num_qubits));
  }

  unsigned TileQubits() const { return partition_.tile_qubits; }

  // ApplyBitPairSwaps moves whole lane groups, so lanes are never remapped.
  unsigned LaneQubits() const { return kLaneQubits; }

  bool Prepare() {
    if (!ValidateThreads()) return false;
    if (param_.inner_threads > 1 && !PinSmtTeamThreads()) return false;
    LogAdaptiveTileSize();
    LogThreadTeams();
    return true;
  }

  void ApplySwaps(const std::vector<QubitSwap>& swaps) {
    ApplyBitPairSwaps(state_data_, partition_.num_state_qubits, kLaneQubits,
                      swaps, param_.num_threads);
  }

  // The proposal's inner loops: for every tile i, apply every fused gate to
  // the tile while it is cache-resident. Tiles or SMT tile teams run in
  // parallel.
  void ExecuteBatch(const std::vector<ExecutableGate>& gates) const {
    if (param_.inner_threads > 1) {
      ExecuteSmtTileTeams(gates);
    } else {
      ExecuteIndependentTiles(gates);
    }
  }

  void Synchronize() const {}  // All work finishes before ExecuteBatch returns.

 private:
  using TilePartition = gate_batch_internal::TilePartition<StateSpace>;
  using SmtTeamBarrier = gate_batch_internal::SmtTeamBarrier;

  // Low amplitude-index bits that select a SIMD lane. The smallest state is
  // one lane group: MinSize(0) = 2 * 2^kLaneQubits floats.
  static inline const unsigned kLaneQubits =
      std::log2(StateSpace::MinSize(0) / 2);

  // Lanes are never remapped, so a gate's qubits must fit above them.
  static unsigned MinTileQubits(unsigned max_fused_size) {
    return kLaneQubits + max_fused_size;
  }

  // Applies every executable gate to one tile. A team splits each
  // gate among its members, who meet at the barrier before the next gate
  // because each gate consumes the preceding gate's output.
  void ExecuteGatesOnTile(const std::vector<ExecutableGate>& gates,
                          int64_t tile, unsigned team_size = 1,
                          unsigned team_thread_id = 0,
                          SmtTeamBarrier* team_barrier = nullptr) const {
    gate_batch_internal::CooperativeFor::Configure(team_size, team_thread_id);
    auto* tile_data = state_data_ + tile * partition_.floats_per_tile;
    auto tile_view =
        SeqStateSpace::Create(tile_data, partition_.tile_qubits);

    for (const ExecutableGate& gate : gates) {
      seq_sim_.ApplyGate(gate.physical_qubits, gate.matrix.data(), tile_view);
      if (team_barrier != nullptr) team_barrier->Wait(team_size);
    }
  }

  // Unit-sized dynamic scheduling balances independent tiles across
  // cores without adding synchronization to the per-gate loop.
  void ExecuteIndependentTiles(
      const std::vector<ExecutableGate>& gates) const {
#pragma omp parallel for schedule(dynamic, 1) num_threads(param_.num_threads)
    for (int64_t tile = 0; tile < partition_.num_tiles; ++tile) {
      ExecuteGatesOnTile(gates, tile);
    }
  }

  // Pins each OpenMP worker to its SMT team CPU once per run. Later parallel
  // regions with the same thread count reuse these pinned workers, because
  // libgomp and libomp keep their thread pool between regions.
  bool PinSmtTeamThreads() const {
    std::atomic<bool> pinned{true};

#pragma omp parallel num_threads(param_.num_threads)
    {
      const auto cpu =
          param_.team_thread_cpus[gate_batch_internal::ParallelThreadId()];
      if (!PinCurrentThreadToCpu(cpu)) {
        pinned.store(false, std::memory_order_relaxed);
      }
    }

    if (!pinned.load(std::memory_order_relaxed)) {
      IO::errorf("qsim_gate_batch: failed to pin an SMT worker to its CPU.\n");
      return false;
    }
    return true;
  }

  // Each team of SMT siblings cooperates on one tile at a time.
  void ExecuteSmtTileTeams(const std::vector<ExecutableGate>& gates) const {
    auto team_barriers =
        std::make_unique<SmtTeamBarrier[]>(param_.num_threads);

#pragma omp parallel num_threads(param_.num_threads)
    {
      const auto assignment = gate_batch_internal::AssignSmtTeam(
          param_.inner_threads, gate_batch_internal::ParallelThreadCount(),
          gate_batch_internal::ParallelThreadId());
      if (assignment.active) {
        for (int64_t tile = assignment.team_id;
             tile < partition_.num_tiles; tile += assignment.num_teams) {
          ExecuteGatesOnTile(gates, tile, assignment.team_size,
                             assignment.lane,
                             &team_barriers[assignment.team_id]);
        }
      }
    }
  }

  bool ValidateThreads() const {
    if (param_.num_threads == 0) {
      IO::errorf("qsim_gate_batch: num_threads must be at least 1.\n");
      return false;
    }
    if (param_.inner_threads <= 1) return true;
    if (!gate_batch_internal::kHasOpenMP) {
      IO::errorf("qsim_gate_batch: SMT teams require OpenMP.\n");
      return false;
    }
    if (param_.num_threads % param_.inner_threads != 0) {
      IO::errorf("qsim_gate_batch: num_threads must be divisible by "
                 "inner_threads.\n");
      return false;
    }
    if (param_.team_thread_cpus.size() != param_.num_threads) {
      IO::errorf("qsim_gate_batch: SMT mode requires one CPU assignment "
                 "per thread.\n");
      return false;
    }
    return true;
  }

  void LogAdaptiveTileSize() const {
    if (verbosity_ > 1 &&
        partition_.tile_qubits != partition_.requested_tile_qubits) {
      IO::messagef("adaptive tile size: L=%u reduced to L=%u, producing "
                   "%lld tiles for %u threads.\n",
                   partition_.requested_tile_qubits, partition_.tile_qubits,
                   static_cast<long long>(partition_.num_tiles),
                   param_.num_threads);
    }
  }

  void LogThreadTeams() const {
    if (verbosity_ <= 1 || param_.inner_threads <= 1) return;
    for (unsigned thread = 0; thread < param_.num_threads;
         thread += param_.inner_threads) {
      const auto team = thread / param_.inner_threads;
      for (unsigned lane = 0; lane < param_.inner_threads; ++lane) {
        IO::messagef("SMT team %u lane %u: CPU %u\n", team, lane,
                     param_.team_thread_cpus[thread + lane]);
      }
    }
  }

  const Parameter& param_;
  const unsigned verbosity_;
  const TilePartition partition_;
  fp_type* const state_data_;
  SeqSimulator seq_sim_;
};

// The CPU gate-batched runner. Keeps qsim's conventional Run signature,
// whose Factory argument is present only for API compatibility.
template <typename IO, typename Fuser, typename Factory>
class QSimGateBatchRunner final
    : public GateBatchRunner<IO, Fuser, CpuGateBatchBackend<IO, Factory>> {
  using Base = GateBatchRunner<IO, Fuser, CpuGateBatchBackend<IO, Factory>>;

 public:
  using typename Base::Parameter;
  using typename Base::QubitMappedState;
  using typename Base::State;

  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, State& state) {
    (void) factory;
    return Base::Run(param, circuit, state);
  }

  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, QubitMappedState& state) {
    (void) factory;
    return Base::Run(param, circuit, state);
  }
};

}  // namespace qsim

#endif  // RUN_QSIM_GATE_BATCH_H_
