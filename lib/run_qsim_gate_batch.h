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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gate_batch_runner.h"
#include "qubit_remap.h"
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

inline unsigned OnlineProcessors() {
#if defined(_SC_NPROCESSORS_ONLN)
  const long count = sysconf(_SC_NPROCESSORS_ONLN);
  if (count > 0) return static_cast<unsigned>(count);
#endif
  return std::max(1u, std::thread::hardware_concurrency());
}

// Running two SMT threads on the same physical core across independent tiles
// doubles the per-core working set and causes L2 cache thrashing. Clamp the
// active worker count to the physical core count when SMT is enabled.
inline unsigned ClampToPhysicalCores(unsigned num_threads) {
  unsigned smt = 0;
  std::ifstream("/sys/devices/system/cpu/smt/active") >> smt;
  const unsigned phy_core_cnt = std::max(1u, OnlineProcessors() >> (smt != 0));
  return std::min(num_threads, phy_core_cnt);
}

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

struct TileFor {
  explicit TileFor(unsigned num_threads) { (void) num_threads; }

  static void Configure(unsigned num_threads, unsigned thread_id) {
    num_threads_ = num_threads;
    thread_id_ = thread_id;
  }

  template <typename Function, typename... Args>
  static void Run(uint64_t size, Function&& func, Args&&... args) {
    const auto num_threads = num_threads_;
    if (num_threads <= 1) {
      for (uint64_t i = 0; i < size; ++i) {
        func(1, 0, i, args...);
      }
      return;
    }
    const auto thread_id = thread_id_;
    const auto begin = size * thread_id / num_threads;
    const auto end = size * (thread_id + 1) / num_threads;
    for (uint64_t i = begin; i < end; ++i) {
      func(num_threads, thread_id, i, args...);
    }
  }

 private:
  inline static thread_local unsigned num_threads_ = 1;
  inline static thread_local unsigned thread_id_ = 0;
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
  using TileSimulator = typename gate_batch_internal::ReplaceFor<
      typename Factory::Simulator, gate_batch_internal::TileFor>::type;
  using TileStateSpace = typename TileSimulator::StateSpace;
  using ExecutableGate = FusedGate<fp_type>;

  struct Parameter {
    unsigned num_threads = 1;
  };

  static constexpr unsigned kDefaultTileQubits = 18;
  static constexpr unsigned kMaxGateQubits = 6;
  // Floor of 5 balances remap budget against swap-pass span length.
  static constexpr unsigned kDefaultEvictionFloor = 5;

  template <typename RunnerParameter>
  CpuGateBatchBackend(const RunnerParameter& param, unsigned num_qubits,
                      State& state)
      : param_(param),
        verbosity_(param.verbosity),
        active_threads_(
            gate_batch_internal::ClampToPhysicalCores(param.num_threads)),
        partition_(num_qubits, param.tile_qubits, active_threads_,
                   MinTileQubits(param.max_fused_size), kDefaultTileQubits),
        state_data_(state.get()),
        tile_sim_(1) {
    assert(partition_.tile_qubits >= std::min(kLaneQubits, num_qubits));
  }

  unsigned TileQubits() const { return partition_.tile_qubits; }
  unsigned RequestedTileQubits() const {
    return partition_.requested_tile_qubits;
  }

  // SIMD lanes are never remapped to preserve lane-group layout.
  unsigned LaneQubits() const { return kLaneQubits; }

  bool Prepare() {
    if (param_.num_threads == 0) {
      IO::errorf("qsim_gate_batch: num_threads must be at least 1.\n");
      return false;
    }
    LogAdaptiveTileSize();
    return true;
  }

  void ApplySwaps(const std::vector<QubitSwap>& swaps) {
    ApplyBitPairSwaps(state_data_, partition_.num_state_qubits, kLaneQubits,
                      swaps, active_threads_);
  }

  void ExecuteBatch(const std::vector<ExecutableGate>& gates) const {
    if (partition_.num_tiles == 1) {
      ExecuteSingleTileBatch(gates);
    } else {
      ExecuteIndependentTiles(gates);
    }
  }

  void Synchronize() const {}

 private:
  using TilePartition = gate_batch_internal::TilePartition<StateSpace>;

  static inline const unsigned kLaneQubits =
      std::log2(StateSpace::MinSize(0) / 2);

  static unsigned MinTileQubits(unsigned max_fused_size) {
    return kLaneQubits + max_fused_size;
  }

  void ExecuteGatesOnTile(const std::vector<ExecutableGate>& gates,
                          int64_t tile) const {
    gate_batch_internal::TileFor::Configure(1, 0);
    auto* tile_data = state_data_ + tile * partition_.floats_per_tile;
    auto tile_view =
        TileStateSpace::Create(tile_data, partition_.tile_qubits);

    for (const ExecutableGate& gate : gates) {
      tile_sim_.ApplyGate(gate.qubits, gate.matrix.data(), tile_view);
    }
  }

  void ExecuteSingleTileBatch(const std::vector<ExecutableGate>& gates) const {
    constexpr unsigned kMinParallelQubits = 14;
    if (!gate_batch_internal::kHasOpenMP || active_threads_ <= 1 ||
        partition_.tile_qubits < kMinParallelQubits) {
      ExecuteGatesOnTile(gates, 0);
      return;
    }

#pragma omp parallel num_threads(active_threads_)
    {
      gate_batch_internal::TileFor::Configure(
          gate_batch_internal::ParallelThreadCount(),
          gate_batch_internal::ParallelThreadId());
      auto full_view =
          TileStateSpace::Create(state_data_, partition_.tile_qubits);
      for (const ExecutableGate& gate : gates) {
        tile_sim_.ApplyGate(gate.qubits, gate.matrix.data(), full_view);
#pragma omp barrier
      }
    }
  }

  void ExecuteIndependentTiles(
      const std::vector<ExecutableGate>& gates) const {
#pragma omp parallel for schedule(dynamic, 1) num_threads(active_threads_)
    for (int64_t tile = 0; tile < partition_.num_tiles; ++tile) {
      ExecuteGatesOnTile(gates, tile);
    }
  }

  void LogAdaptiveTileSize() const {
    if (verbosity_ > 1 &&
        partition_.tile_qubits != partition_.requested_tile_qubits) {
      IO::messagef("adaptive tile size: L=%u reduced to L=%u, producing "
                   "%lld tiles for %u threads.\n",
                   partition_.requested_tile_qubits, partition_.tile_qubits,
                   static_cast<long long>(partition_.num_tiles),
                   active_threads_);
    }
  }

  const Parameter& param_;
  const unsigned verbosity_;
  const unsigned active_threads_;
  const TilePartition partition_;
  fp_type* const state_data_;
  TileSimulator tile_sim_;
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
