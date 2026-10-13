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

// CPU backend and entry point for the gate-batched runner.

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
#include "seqfor.h"
#include "util.h"

namespace qsim {

namespace gate_batch_internal {

#ifdef _OPENMP
inline constexpr bool kHasOpenMP = true;
inline unsigned ParallelThreadCount() { return omp_get_num_threads(); }
inline unsigned ParallelThreadId() { return omp_get_thread_num(); }
#else
inline constexpr bool kHasOpenMP = false;
inline unsigned ParallelThreadCount() { return 1; }
inline unsigned ParallelThreadId() { return 0; }
#endif

template <typename StateSpace>
struct TilePartition {
  static constexpr unsigned kCacheResidentQubits = 26;

  TilePartition(unsigned state_qubits, unsigned requested_tile_qubits,
                unsigned num_threads, unsigned min_tile_qubits,
                unsigned default_tile_qubits)
      : num_state_qubits(state_qubits),
        requested_tile_qubits(
            std::min(requested_tile_qubits, state_qubits)),
        tile_qubits(ChooseTileQubits(
            state_qubits, requested_tile_qubits, num_threads,
            min_tile_qubits, default_tile_qubits)),
        floats_per_tile(StateSpace::MinSize(tile_qubits)),
        num_tiles(int64_t{1} << (state_qubits - tile_qubits)) {}

  // Keep LLC-resident states whole unless a smaller tile was explicitly requested.
  static unsigned ChooseTileQubits(unsigned state_qubits,
                                   unsigned requested_tile_qubits,
                                   unsigned num_threads,
                                   unsigned min_tile_qubits,
                                   unsigned default_tile_qubits) {
    if (state_qubits < kCacheResidentQubits &&
        requested_tile_qubits >= std::min(state_qubits, default_tile_qubits)) {
      return state_qubits;
    }
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

  unsigned num_state_qubits;
  unsigned requested_tile_qubits;
  unsigned tile_qubits;
  uint64_t floats_per_tile;
  int64_t num_tiles;
};

// SMT sibling barrier; keep barriers on separate cache lines.
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

struct SmtTeamAssignment {
  unsigned team_size;
  unsigned num_teams;
  unsigned team_id;
  unsigned lane;
  bool active;
};

inline SmtTeamAssignment AssignSmtTeam(unsigned inner_threads,
                                       unsigned num_threads,
                                       unsigned thread_id) {
  const auto team_size = std::max(1u, std::min(inner_threads, num_threads));
  const auto num_teams = num_threads / team_size;
  const auto team_id = thread_id / team_size;
  return {team_size, num_teams, team_id, thread_id % team_size,
          team_id < num_teams};
}

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
  using SeqSimulator = typename gate_batch_internal::ReplaceFor<
      typename Factory::Simulator, gate_batch_internal::CooperativeFor>::type;
  using SeqStateSpace = typename SeqSimulator::StateSpace;
  using SingleThreadSimulator = typename gate_batch_internal::ReplaceFor<
      typename Factory::Simulator, SequentialFor>::type;
  using SingleThreadStateSpace = typename SingleThreadSimulator::StateSpace;
  using ExecutableGate = gate_batch_internal::ExecutableGate<fp_type>;

  struct Parameter {
    unsigned num_threads = 1;
    unsigned inner_threads = 1;
    std::vector<unsigned> team_thread_cpus;
  };

  static constexpr unsigned kDefaultTileQubits = 19;
  static constexpr unsigned kMaxGateQubits = 6;
  // Floor of 5 balances remap budget against swap-pass span length.
  static constexpr unsigned kDefaultEvictionFloor = 5;

  template <typename RunnerParameter>
  CpuGateBatchBackend(const RunnerParameter& param, unsigned num_qubits,
                      State& state)
      : param_(param),
        verbosity_(param.verbosity),
        partition_(num_qubits, param.tile_qubits, param.num_threads,
                   MinTileQubits(param.max_fused_size), kDefaultTileQubits),
        state_data_(state.get()),
        seq_sim_(1),
        single_thread_sim_(1) {
    assert(partition_.tile_qubits >= std::min(kLaneQubits, num_qubits));
  }

  unsigned TileQubits() const { return partition_.tile_qubits; }
  unsigned RequestedTileQubits() const {
    return partition_.requested_tile_qubits;
  }

  // SIMD lanes are never remapped to preserve lane-group layout.
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

  void ExecuteBatch(const std::vector<ExecutableGate>& gates) const {
    if (partition_.num_tiles == 1) {
      ExecuteSingleTileBatch(gates);
    } else if (param_.inner_threads > 1) {
      ExecuteSmtTileTeams(gates);
    } else {
      ExecuteIndependentTiles(gates);
    }
  }

  void Synchronize() const {}

 private:
  using TilePartition = gate_batch_internal::TilePartition<StateSpace>;
  using SmtTeamBarrier = gate_batch_internal::SmtTeamBarrier;

  static inline const unsigned kLaneQubits =
      std::log2(StateSpace::MinSize(0) / 2);

  static unsigned MinTileQubits(unsigned max_fused_size) {
    return kLaneQubits + max_fused_size;
  }

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

  void ExecuteSingleTileBatch(const std::vector<ExecutableGate>& gates) const {
    constexpr unsigned kMinParallelQubits = 14;
    if (!gate_batch_internal::kHasOpenMP || param_.num_threads <= 1 ||
        partition_.tile_qubits < kMinParallelQubits) {
      auto full_view =
          SingleThreadStateSpace::Create(state_data_, partition_.tile_qubits);
      for (const ExecutableGate& gate : gates) {
        single_thread_sim_.ApplyGate(gate.physical_qubits, gate.matrix.data(),
                                     full_view);
      }
      return;
    }

#pragma omp parallel num_threads(param_.num_threads)
    {
      gate_batch_internal::CooperativeFor::Configure(
          gate_batch_internal::ParallelThreadCount(),
          gate_batch_internal::ParallelThreadId());
      auto full_view =
          SeqStateSpace::Create(state_data_, partition_.tile_qubits);
      for (const ExecutableGate& gate : gates) {
        seq_sim_.ApplyGate(gate.physical_qubits, gate.matrix.data(), full_view);
#pragma omp barrier
      }
    }
  }

  void ExecuteIndependentTiles(
      const std::vector<ExecutableGate>& gates) const {
#pragma omp parallel for schedule(dynamic, 1) num_threads(param_.num_threads)
    for (int64_t tile = 0; tile < partition_.num_tiles; ++tile) {
      auto* tile_data = state_data_ + tile * partition_.floats_per_tile;
      auto tile_view =
          SingleThreadStateSpace::Create(tile_data, partition_.tile_qubits);
      for (const ExecutableGate& gate : gates) {
        single_thread_sim_.ApplyGate(gate.physical_qubits, gate.matrix.data(),
                                     tile_view);
      }
    }
  }

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
  SingleThreadSimulator single_thread_sim_;
};

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
