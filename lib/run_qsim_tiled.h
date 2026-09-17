#ifndef QSIM_RUN_QSIM_TILED_H_
#define QSIM_RUN_QSIM_TILED_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <complex>
#include <cmath>
#include <random>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpu_topology.h"
#include "gate_batch_planner.h"
#include "run_qsim.h"
#include "tile_partition.h"
#include "tile_scheduler.h"
#include "qubit_remap.h"
#include "statespace_basic.h"
#include "seqfor.h"

namespace qsim {

struct TiledOptions {
  unsigned tile_qubits = 0;
  unsigned local_qubits = 0;
  unsigned outer_threads = 1;
  unsigned inner_threads = 1;
  unsigned max_fused_size = 2;
  TileSchedule schedule = TileSchedule::kLinear;
  bool numa = true;
};

struct TiledStats {
  uint64_t tiles = 0, batches = 0;
  double planning_seconds = 0;
  double allocation_seconds = 0, remap_seconds = 0, fusion_seconds = 0;
  double gate_seconds = 0, scheduling_seconds = 0, total_seconds = 0;
  CpuTopology topology;
  std::vector<std::complex<float>> first_amplitudes;
  double norm = 0;
  std::string fallback_reason;
  std::string backend;
};

namespace detail {

template <typename Function>
inline void ParallelTiles(uint64_t tiles, unsigned workers,
                          const std::vector<unsigned>& affinity,
                          TileSchedule schedule, Function function,
                          const CpuTopology* topology = nullptr,
                          double* scheduling_seconds = nullptr,
                          unsigned team_threads = 1) {
  workers = std::max(1U, std::min<unsigned>(workers, tiles ? tiles : 1));
  TileScheduler scheduler(tiles, workers, schedule);
  static const CpuTopology detected_topology = CpuTopology::Detect();
  CpuTopology placement = topology ? *topology : detected_topology;
  if (!affinity.empty()) placement.affinity = affinity;
  const std::vector<unsigned> worker_cpus = scheduler.WorkerCpus(placement);
  const std::vector<std::vector<unsigned>> worker_teams =
      scheduler.WorkerTeams(placement, team_threads);
  if (workers == 1) {
    for (uint64_t tile = 0; tile < tiles; ++tile) function(tile, 0);
    return;
  }
  std::vector<std::thread> threads;
  std::vector<double> worker_seconds(workers, 0);
  const auto scheduling_start = std::chrono::steady_clock::now();
  DynamicTileQueue dynamic_queue(tiles);
  for (unsigned worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      if (team_threads > 1) {
        TileScheduler::PinCurrentThread(worker_teams, worker);
      } else {
        TileScheduler::PinCurrentThread(worker_cpus, worker);
      }
      auto worker_start = std::chrono::steady_clock::now();
      if (scheduler.dynamic()) {
        uint64_t tile;
        while (dynamic_queue.Next(&tile)) function(tile, worker);
      } else {
        for (uint64_t tile = 0; tile < tiles; ++tile)
          if (scheduler.WorkerFor(tile) == worker) function(tile, worker);
      }
      worker_seconds[worker] = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - worker_start).count();
    });
  }
  for (auto& thread : threads) thread.join();
  if (scheduling_seconds) {
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scheduling_start).count();
    const double busy = *std::max_element(worker_seconds.begin(), worker_seconds.end());
    *scheduling_seconds += std::max(0.0, wall - busy);
  }
}

template <typename FP>
inline std::complex<FP> MatrixElement(const std::vector<FP>& m,
                                      unsigned n, unsigned row, unsigned col) {
  return {m[2 * (n * row + col)], m[2 * (n * row + col) + 1]};
}

template <typename FP>
inline void ApplyNormalMatrix(const std::vector<FP>& matrix,
                              const std::vector<unsigned>& qs,
                              FP* state, unsigned num_qubits,
                              uint64_t first, uint64_t count) {
  if (qs.size() > 6 || matrix.empty()) return;
  const unsigned qsize = 1U << qs.size();
  if (qs.empty()) {
    auto scale = MatrixElement(matrix, 1, 0, 0);
    for (uint64_t i = first; i < first + count; ++i) {
      std::complex<FP> value{state[2 * i], state[2 * i + 1]};
      value *= scale;
      state[2 * i] = value.real(); state[2 * i + 1] = value.imag();
    }
    return;
  }
  std::array<uint64_t, 6> masks{};
  for (unsigned i = 0; i < qs.size(); ++i) masks[i] = uint64_t{1} << qs[i];
  const uint64_t end = first + count;
  for (uint64_t base = first; base < end; ++base) {
    if ((base & masks[0]) != 0) continue;
    bool zero = true;
    for (unsigned q = 1; q < qs.size(); ++q) if (base & masks[q]) zero = false;
    if (!zero) continue;
    std::array<std::complex<FP>, 64> in{}, out{};
    for (unsigned col = 0; col < qsize; ++col) {
      uint64_t index = base;
      for (unsigned q = 0; q < qs.size(); ++q) if (col & (1U << q)) index |= masks[q];
      in[col] = {state[2 * index], state[2 * index + 1]};
    }
    for (unsigned row = 0; row < qsize; ++row)
      for (unsigned col = 0; col < qsize; ++col)
        out[row] += MatrixElement(matrix, qsize, row, col) * in[col];
    for (unsigned row = 0; row < qsize; ++row) {
      uint64_t index = base;
      for (unsigned q = 0; q < qs.size(); ++q) if (row & (1U << q)) index |= masks[q];
      state[2 * index] = out[row].real(); state[2 * index + 1] = out[row].imag();
    }
  }
}

template <typename FP>
inline void ApplyNormalMatrixCooperative(const std::vector<FP>& matrix,
                                         const std::vector<unsigned>& qs,
                                         FP* state, unsigned num_qubits,
                                         uint64_t first, uint64_t count,
                                         unsigned inner_threads) {
  if (inner_threads <= 1 || qs.empty()) {
    ApplyNormalMatrix(matrix, qs, state, num_qubits, first, count);
    return;
  }
  std::vector<uint64_t> representatives;
  std::array<uint64_t, 6> masks{};
  for (unsigned i = 0; i < qs.size(); ++i) masks[i] = uint64_t{1} << qs[i];
  for (uint64_t base = first; base < first + count; ++base) {
    bool zero = true;
    for (auto mask : masks) if (base & mask) zero = false;
    if (zero) representatives.push_back(base);
  }
  if (representatives.size() < 2) {
    ApplyNormalMatrix(matrix, qs, state, num_qubits, first, count);
    return;
  }
  inner_threads = std::min<unsigned>(inner_threads, representatives.size());
  std::vector<std::thread> threads;
  for (unsigned worker = 0; worker < inner_threads; ++worker) {
    threads.emplace_back([&, worker] {
      for (uint64_t i = worker; i < representatives.size(); i += inner_threads)
        ApplyNormalMatrix(matrix, qs, state, num_qubits, representatives[i], 1);
    });
  }
  for (auto& thread : threads) thread.join();
}

template <typename FP>
inline void ApplyNormalControlled(const Gate<FP>& gate,
                                  const std::vector<unsigned>& controls,
                                  uint64_t values, FP* state,
                                  unsigned num_qubits, uint64_t first,
                                  uint64_t count) {
  if (controls.size() > 63) return;
  uint64_t mask = 0, bits = 0;
  for (unsigned i = 0; i < controls.size(); ++i) {
    mask |= uint64_t{1} << controls[i];
    if (values & (uint64_t{1} << i)) bits |= uint64_t{1} << controls[i];
  }
  const uint64_t end = first + count;
  uint64_t target_mask = 0;
  for (auto q : gate.qubits) target_mask |= uint64_t{1} << q;
  const uint64_t stride = uint64_t{1} << gate.qubits.size();
  (void)stride;
  for (uint64_t i = first; i < end; ++i)
    if ((i & mask) == bits) {
      // ApplyNormalMatrix will visit the complete target subspace. Only its
      // zero target representative is needed for each control subspace.
      if ((i & target_mask) == 0)
        ApplyNormalMatrix(gate.matrix, gate.qubits, state, num_qubits, i, 1);
    }
}

template <typename FP>
inline void RemapQubits(std::vector<unsigned>& qs, const QubitLayout& layout) {
  for (auto& q : qs) q = layout.LogicalToPhysical(q);
}

template <typename FP>
inline void NormalizeMappedGate(std::vector<unsigned>& qs,
                                std::vector<FP>& matrix) {
  auto permutation = NormalToGateOrderPermutation(qs);
  if (!permutation.empty()) MatrixShuffle(permutation, qs.size(), matrix);
  std::sort(qs.begin(), qs.end());
}

inline void NormalizeMappedControls(std::vector<unsigned>& controls,
                                    uint64_t& values) {
  std::vector<std::pair<unsigned, unsigned>> mapped;
  mapped.reserve(controls.size());
  for (unsigned i = 0; i < controls.size(); ++i)
    mapped.push_back({controls[i], static_cast<unsigned>((values >> i) & 1)});
  std::sort(mapped.begin(), mapped.end());
  values = 0;
  for (unsigned i = 0; i < mapped.size(); ++i) {
    controls[i] = mapped[i].first;
    values |= uint64_t{mapped[i].second} << i;
  }
}

template <typename StateSpace>
inline void ApplyTiledBitSwap(
    std::vector<typename StateSpace::State>& states, unsigned tile_qubits,
    unsigned a, unsigned b, unsigned workers = 1,
    const std::vector<unsigned>& affinity = {},
    TileSchedule schedule = TileSchedule::kLinear) {
  if (a == b) return;
  const uint64_t tiles = states.size();
  const uint64_t per_tile = uint64_t{1} << tile_qubits;
  if (a < tile_qubits && b < tile_qubits) {
    const uint64_t ma = uint64_t{1} << a;
    const uint64_t mb = uint64_t{1} << b;
    ParallelTiles(states.size(), workers, affinity, schedule,
                  [&](uint64_t tile, unsigned) {
      auto& state = states[tile];
      for (uint64_t base = 0; base < per_tile; ++base) {
        if ((base & (ma | mb)) != 0) continue;
        auto x = StateSpace::GetAmpl(state, base | ma);
        auto y = StateSpace::GetAmpl(state, base | mb);
        StateSpace::SetAmpl(state, base | ma, y);
        StateSpace::SetAmpl(state, base | mb, x);
      }
    });
    return;
  }
  if (a >= tile_qubits && b >= tile_qubits) {
    const uint64_t ma = uint64_t{1} << (a - tile_qubits);
    const uint64_t mb = uint64_t{1} << (b - tile_qubits);
    ParallelTiles(tiles, workers, affinity, schedule,
                  [&](uint64_t tile, unsigned) {
      if ((tile & (ma | mb)) != 0) return;
      std::swap(states[tile | ma], states[tile | mb]);
                  });
    return;
  }
  unsigned high = a >= tile_qubits ? a : b;
  unsigned low = a >= tile_qubits ? b : a;
  const uint64_t high_mask = uint64_t{1} << (high - tile_qubits);
  const uint64_t low_mask = uint64_t{1} << low;
  ParallelTiles(tiles, workers, affinity, schedule,
                [&](uint64_t tile, unsigned) {
    if (tile & high_mask) return;
    uint64_t other = tile | high_mask;
    for (uint64_t base = 0; base < per_tile; ++base) {
      if (base & low_mask) continue;
      auto x = StateSpace::GetAmpl(states[tile], base | low_mask);
      auto y = StateSpace::GetAmpl(states[other], base);
      StateSpace::SetAmpl(states[tile], base | low_mask, y);
      StateSpace::SetAmpl(states[other], base, x);
    }
                });
}

template <typename StateSpace>
inline bool ApplyTiledMeasurement(
    std::vector<typename StateSpace::State>& states, unsigned tile_qubits,
    const std::vector<unsigned>& qubits, std::mt19937& random) {
  using FP = typename StateSpace::fp_type;
  if (qubits.size() > tile_qubits || qubits.size() >= 63) return false;
  const uint64_t outcomes = uint64_t{1} << qubits.size();
  const uint64_t per_tile = uint64_t{1} << tile_qubits;
  std::vector<double> probabilities(outcomes, 0);
  double total = 0;
  for (const auto& state : states) {
    for (uint64_t index = 0; index < per_tile; ++index) {
      uint64_t outcome = 0;
      for (unsigned bit = 0; bit < qubits.size(); ++bit)
        outcome |= ((index >> qubits[bit]) & 1) << bit;
      auto amplitude = StateSpace::GetAmpl(state, index);
      probabilities[outcome] += std::norm(amplitude);
    }
  }
  for (auto probability : probabilities) total += probability;
  if (!(total > 0)) return false;
  std::uniform_real_distribution<double> distribution(0, total);
  double sample = distribution(random), cumulative = 0;
  uint64_t selected = outcomes - 1;
  for (uint64_t outcome = 0; outcome < outcomes; ++outcome) {
    cumulative += probabilities[outcome];
    if (sample <= cumulative) { selected = outcome; break; }
  }
  const double selected_probability = probabilities[selected];
  if (!(selected_probability > 0)) return false;
  const FP scale = static_cast<FP>(1 / std::sqrt(selected_probability));
  for (auto& state : states) {
    for (uint64_t index = 0; index < per_tile; ++index) {
      uint64_t outcome = 0;
      for (unsigned bit = 0; bit < qubits.size(); ++bit)
        outcome |= ((index >> qubits[bit]) & 1) << bit;
      if (outcome != selected) {
        StateSpace::SetAmpl(state, index, FP{0}, FP{0});
      } else {
        auto amplitude = StateSpace::GetAmpl(state, index) * scale;
        StateSpace::SetAmpl(state, index, amplitude);
      }
    }
  }
  return true;
}

template <typename FP, typename Operation, typename FusedOperation,
          typename Simulator, typename StateSpace>
inline bool ApplySimdFused(
    const FusedOperation& operation, const QubitLayout& layout,
    std::vector<typename StateSpace::State>& states, unsigned tile_qubits,
    unsigned workers, const std::vector<unsigned>& affinity,
    TileSchedule schedule, std::mt19937& random, unsigned inner_threads,
    double* scheduling_seconds = nullptr) {
  if (const auto* measurement = OpGetAlternative<Measurement>(operation)) {
    auto qs = measurement->qubits;
    RemapQubits<FP>(qs, layout);
    for (auto q : qs) if (q >= tile_qubits) return false;
    return ApplyTiledMeasurement<StateSpace>(states, tile_qubits, qs, random);
  }
  if (const auto* fused = OpGetAlternative<FusedGate<FP>>(operation)) {
    if (fused->qubits.size() > 6 || fused->matrix.empty()) return false;
    auto qs = fused->qubits;
    auto matrix = fused->matrix;
    RemapQubits<FP>(qs, layout);
    for (auto q : qs) if (q >= tile_qubits) return false;
    NormalizeMappedGate(qs, matrix);
    ParallelTiles(states.size(), workers, affinity, schedule,
                  [&](uint64_t tile, unsigned) {
      Simulator simulator(std::max(1U, inner_threads));
      simulator.ApplyGate(qs, matrix.data(), states[tile]);
    }, nullptr, scheduling_seconds, inner_threads);
    return true;
  }
  if (const auto* original = std::get_if<const Operation*>(&operation)) {
    if (*original == nullptr) return false;
    if (const auto* gate = OpGetAlternative<Gate<FP>>(**original)) {
      auto copy = *gate;
      RemapQubits<FP>(copy.qubits, layout);
      for (auto q : copy.qubits) if (q >= tile_qubits) return false;
      NormalizeMappedGate(copy.qubits, copy.matrix);
      ParallelTiles(states.size(), workers, affinity, schedule,
                    [&](uint64_t tile, unsigned) {
        Simulator simulator(std::max(1U, inner_threads));
        simulator.ApplyGate(copy.qubits, copy.matrix.data(), states[tile]);
      }, nullptr, scheduling_seconds, inner_threads);
      return true;
    }
    if (const auto* controlled = OpGetAlternative<ControlledGate<FP>>(**original)) {
      if (controlled->qubits.size() > 6 || controlled->matrix.empty()) return false;
      auto copy = *controlled;
      RemapQubits<FP>(copy.qubits, layout);
      RemapQubits<FP>(copy.controlled_by, layout);
      NormalizeMappedControls(copy.controlled_by, copy.cmask);
      for (auto q : copy.qubits) if (q >= tile_qubits) return false;
      for (auto q : copy.controlled_by) if (q >= tile_qubits) return false;
      NormalizeMappedGate(copy.qubits, copy.matrix);
      ParallelTiles(states.size(), workers, affinity, schedule,
                    [&](uint64_t tile, unsigned) {
        Simulator simulator(std::max(1U, inner_threads));
        simulator.ApplyControlledGate(copy.qubits, copy.controlled_by,
                                      copy.cmask, copy.matrix.data(),
                                      states[tile]);
                    }, nullptr, scheduling_seconds, inner_threads);
      return true;
    }
  }
  return false;
}

template <typename FP>
inline bool ApplyNormalOperation(const Gate<FP>& gate, const QubitLayout& layout,
                                 FP* state, unsigned n, unsigned tile_q,
                                 uint64_t tiles, unsigned workers,
                                 const std::vector<unsigned>& affinity,
                                 TileSchedule schedule, unsigned inner_threads) {
  if (gate.qubits.size() > 6 || gate.matrix.empty()) return false;
  auto qs = gate.qubits; RemapQubits<FP>(qs, layout);
  for (auto q : qs) if (q >= tile_q) return false;
  uint64_t per_tile = uint64_t{1} << tile_q;
  ParallelTiles(tiles, workers, affinity, schedule, [&](uint64_t tile, unsigned) {
    ApplyNormalMatrixCooperative(gate.matrix, qs, state, n, tile * per_tile,
                                 per_tile, inner_threads);
  });
  return true;
}

template <typename FP>
inline bool ApplyNormalOperation(const ControlledGate<FP>& gate,
                                 const QubitLayout& layout, FP* state,
                                 unsigned n, unsigned tile_q, uint64_t tiles,
                                 unsigned workers, const std::vector<unsigned>& affinity,
                                 TileSchedule schedule, unsigned = 1) {
  if (gate.qubits.size() > 6 || gate.matrix.empty()) return false;
  auto target = gate.qubits; auto controls = gate.controlled_by;
  RemapQubits<FP>(target, layout); RemapQubits<FP>(controls, layout);
  for (auto q : target) if (q >= tile_q) return false;
  for (auto q : controls) if (q >= tile_q) return false;
  uint64_t per_tile = uint64_t{1} << tile_q;
  Gate<FP> copy = gate; copy.qubits = target;
  ParallelTiles(tiles, workers, affinity, schedule, [&](uint64_t tile, unsigned) {
    ApplyNormalControlled(copy, controls, gate.cmask, state, n,
                          tile * per_tile, per_tile);
  });
  return true;
}

template <typename FP>
inline bool ApplyNormalOperation(const FusedGate<FP>& gate,
                                 const QubitLayout& layout, FP* state,
                                 unsigned n, unsigned tile_q, uint64_t tiles,
                                 unsigned workers, const std::vector<unsigned>& affinity,
                                 TileSchedule schedule, unsigned inner_threads) {
  if (gate.qubits.size() > 6 || gate.matrix.empty()) return false;
  auto qs = gate.qubits; RemapQubits<FP>(qs, layout);
  for (auto q : qs) if (q >= tile_q) return false;
  uint64_t per_tile = uint64_t{1} << tile_q;
  ParallelTiles(tiles, workers, affinity, schedule, [&](uint64_t tile, unsigned) {
    ApplyNormalMatrixCooperative(gate.matrix, qs, state, n, tile * per_tile,
                                 per_tile, inner_threads);
  });
  return true;
}

template <typename FP>
inline bool ApplyNormalOperation(const Measurement& measurement,
                                 const QubitLayout& layout, FP* state,
                                 unsigned n, unsigned tile_q, uint64_t tiles,
                                 std::mt19937& random, unsigned workers = 1,
                                 const std::vector<unsigned>& affinity = {},
                                 TileSchedule schedule = TileSchedule::kLinear) {
  auto qs = measurement.qubits; RemapQubits<FP>(qs, layout);
  for (auto q : qs) if (q >= tile_q) return false;
  using SS = StateSpaceBasic<SequentialFor, FP>;
  SS space(1); auto wrapped = SS::Create(state, n);
  auto result = space.Measure(qs, random, wrapped);
  return result.valid;
}

template <typename FP, typename Op>
inline bool ApplyNormalOperation(const Op& op, const QubitLayout& layout,
                                 FP*, unsigned, unsigned, uint64_t, std::mt19937&,
                                 unsigned = 1, const std::vector<unsigned>& = {},
                                 TileSchedule = TileSchedule::kLinear) {
  (void)op; (void)layout; return false;
}

template <typename FP, typename Operation, typename FusedOperation>
inline bool ApplyNormalFused(const FusedOperation& operation,
                             const QubitLayout& layout, FP* state,
                             unsigned n, unsigned tile_q, uint64_t tiles,
                             unsigned workers, const std::vector<unsigned>& affinity,
                             TileSchedule schedule, std::mt19937& random,
                             unsigned inner_threads) {
  if (const auto* gate = OpGetAlternative<FusedGate<FP>>(operation))
  {
    // Applying the original components avoids the O(2^(2k)) dense multiply
    // for common one- and two-qubit fused blocks while preserving the exact
    // chronological order used by CalculateFusedMatrix.
    if (!gate->gates.empty()) {
      for (const auto& component : gate->gates) {
        const auto* component_gate = std::get_if<const Gate<FP>*>(&component);
        if (component_gate == nullptr || *component_gate == nullptr) return false;
        if (!ApplyNormalOperation(**component_gate, layout, state, n, tile_q,
                                  tiles, workers, affinity, schedule,
                                  inner_threads)) return false;
      }
      return true;
    }
    return ApplyNormalOperation(*gate, layout, state, n, tile_q, tiles,
                                workers, affinity, schedule, inner_threads);
  }
  if (const auto* measurement = OpGetAlternative<Measurement>(operation))
    return ApplyNormalOperation(*measurement, layout, state, n, tile_q, tiles,
                                random, workers, affinity, schedule);
  if (const auto* original = std::get_if<const Operation*>(&operation)) {
    if (const auto* gate = OpGetAlternative<Gate<FP>>(**original))
      return ApplyNormalOperation(*gate, layout, state, n, tile_q, tiles,
                                  workers, affinity, schedule, inner_threads);
    if (const auto* gate = OpGetAlternative<ControlledGate<FP>>(**original))
      return ApplyNormalOperation(*gate, layout, state, n, tile_q, tiles,
                                  workers, affinity, schedule, inner_threads);
  }
  return false;
}

}  // namespace detail

// Executes a float circuit in normal interleaved order.  This is the portable
// tile kernel path: after each batch's disjoint swaps, every gate in the
// batch touches only tile-local low bits, so tiles can be processed in
// parallel without cross-tile writes.
template <typename IO, typename Fuser, typename Circuit>
bool RunQSimTiledNormal(const TiledOptions& options, const Circuit& circuit,
                        TiledStats* stats = nullptr) {
  auto total_start = std::chrono::steady_clock::now();
  const auto topology = CpuTopology::Detect();
  const auto& ops = Operations<Circuit>::get(circuit);
  using Operation = typename std::decay_t<decltype(ops)>::value_type;
  using FP = OpFpType<Operation>;
  static_assert(std::is_same_v<FP, float>, "normal tiled runner currently uses float amplitudes");
  unsigned tile_q = options.tile_qubits;
  if (tile_q == 0) {
    tile_q = 1;
    while (tile_q < circuit.num_qubits &&
           (uint64_t{1} << (tile_q + 1)) * sizeof(FP) * 2 <= topology.TargetTileCacheBytes()) ++tile_q;
  }
  tile_q = std::min(tile_q, circuit.num_qubits);
  TilePartition partition(circuit.num_qubits, tile_q);
  auto planning_start = std::chrono::steady_clock::now();
  GateBatchPlanner<Operation> planner(ops, options.local_qubits
      ? options.local_qubits : tile_q, options.max_fused_size);
  auto batches = planner.Plan();
  auto planning_end = std::chrono::steady_clock::now();

  // Unsupported operation kinds are handled by the SIMD fallback in the
  // generic entry point before any tile state is modified.
  for (const auto& op : ops) {
    if (!OpGetAlternative<Gate<FP>>(op) &&
        !OpGetAlternative<ControlledGate<FP>>(op) &&
        !OpGetAlternative<Measurement>(op)) {
      if (stats) stats->fallback_reason = "unsupported operation";
      return false;
    }
  }

  using StateSpace = StateSpaceBasic<SequentialFor, FP>;
  StateSpace state_space(1);
  auto allocation_start = std::chrono::steady_clock::now();
  auto state = options.numa
      ? state_space.CreateNuma(circuit.num_qubits, topology.numa_nodes)
      : state_space.Create(circuit.num_qubits);
  const double allocation_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - allocation_start).count();
  if (state_space.IsNull(state)) {
    if (stats) stats->fallback_reason = "state allocation";
    return false;
  }
  state_space.SetStateZero(state);

  QubitRemapper remapper(circuit.num_qubits);
  std::vector<QubitSwap> swaps;
  std::mt19937 random(1);
  typename Fuser::Parameter param;
  param.max_fused_size = options.max_fused_size;
  param.verbosity = 0;
  const unsigned workers = std::max(1U, options.outer_threads);
  double remap_seconds = 0, fusion_seconds = 0, gate_seconds = 0;
  for (const auto& batch : batches) {
    if (batch.qubits > tile_q) {
      if (stats) stats->fallback_reason = "batch exceeds tile-local qubits (" +
          std::to_string(batch.qubits) + ">" + std::to_string(tile_q) + ")";
      return false;
    }
    auto remap_start = std::chrono::steady_clock::now();
    auto batch_swaps = remapper.MakeLocal(batch.logical_qubits, tile_q);
    for (const auto& swap : batch_swaps) {
      ApplyBitSwap(state.get(), circuit.num_qubits, swap.physical_a, swap.physical_b);
      swaps.push_back(swap);
    }
    remap_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - remap_start).count();
    auto fusion_start = std::chrono::steady_clock::now();
    std::vector<Operation> batch_ops;
    batch_ops.reserve(batch.indices.size());
    for (auto index : batch.indices) batch_ops.push_back(ops[index]);
    auto fused = Fuser::template FuseGates<Operation>(
        param, circuit.num_qubits, batch_ops, true);
    fusion_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fusion_start).count();
    auto gate_start = std::chrono::steady_clock::now();
    for (const auto& operation : fused) {
      if (!detail::ApplyNormalFused<FP, Operation>(
              operation, remapper.layout(), state.get(), circuit.num_qubits,
              tile_q, partition.num_tiles(), workers, topology.affinity,
              options.schedule, random, options.inner_threads)) {
        if (stats) stats->fallback_reason = "unsupported fused operation";
        return false;
      }
    }
    gate_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gate_start).count();
  }
  for (auto it = swaps.rbegin(); it != swaps.rend(); ++it)
    ApplyBitSwap(state.get(), circuit.num_qubits, it->physical_a, it->physical_b);
  if (stats) {
    stats->tiles = partition.num_tiles(); stats->batches = batches.size();
    stats->topology = topology;
    stats->backend = "portable-tile";
    stats->planning_seconds = std::chrono::duration<double>(planning_end - planning_start).count();
    stats->allocation_seconds = allocation_seconds;
    stats->remap_seconds = remap_seconds; stats->fusion_seconds = fusion_seconds;
    stats->gate_seconds = gate_seconds;
    stats->total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    const uint64_t count = std::min<uint64_t>(8, uint64_t{1} << circuit.num_qubits);
    stats->first_amplitudes.reserve(count);
    for (uint64_t i = 0; i < count; ++i)
      stats->first_amplitudes.emplace_back(state.get()[2 * i], state.get()[2 * i + 1]);
    for (uint64_t i = 0; i < (uint64_t{1} << circuit.num_qubits); ++i)
      stats->norm += state.get()[2 * i] * state.get()[2 * i] + state.get()[2 * i + 1] * state.get()[2 * i + 1];
  }
  return true;
}

// Uses the selected qsim SIMD StateSpace independently for each tile when no
// remap is required. Each tile is a genuine tile-sized SIMD state, rather than
// a range of a scalar full-state loop. Measurements and nonlocal operations
// are excluded because they require cross-tile coordination/remapping.
template <typename IO, typename Fuser, typename Factory, typename Circuit>
bool RunQSimTiledSimd(const TiledOptions& options, const Factory& factory,
                      const Circuit& circuit, TiledStats* stats = nullptr) {
  const auto topology = CpuTopology::Detect();
  const auto& ops = Operations<Circuit>::get(circuit);
  using Operation = typename std::decay_t<decltype(ops)>::value_type;
  using FP = OpFpType<Operation>;
  unsigned tile_q = options.tile_qubits;
  if (tile_q == 0) {
    tile_q = 1;
    while (tile_q < circuit.num_qubits &&
           (uint64_t{1} << (tile_q + 1)) * sizeof(FP) * 2 <= topology.TargetTileCacheBytes()) ++tile_q;
  }
  tile_q = std::min(tile_q, circuit.num_qubits);
  for (const auto& op : ops) {
    if (OpGetAlternative<Measurement>(op)) return false;
    const auto* gate = OpGetAlternative<Gate<FP>>(op);
    const auto* controlled = OpGetAlternative<ControlledGate<FP>>(op);
    if (gate == nullptr && controlled == nullptr) return false;
    const auto& targets = gate ? gate->qubits : controlled->qubits;
    for (auto q : targets) if (q >= tile_q) return false;
    if (controlled) for (auto q : controlled->controlled_by) if (q >= tile_q) return false;
  }

  TilePartition partition(circuit.num_qubits, tile_q);
  auto planning_start = std::chrono::steady_clock::now();
  GateBatchPlanner<Operation> planner(ops, tile_q, options.max_fused_size);
  auto batches = planner.Plan();
  auto planning_end = std::chrono::steady_clock::now();
  Circuit tile_circuit = circuit;
  tile_circuit.num_qubits = tile_q;
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  std::atomic<bool> ok{true};
  std::vector<double> norms(partition.num_tiles(), 0);
  std::vector<std::complex<FP>> first;
  std::mutex first_mutex;
  auto total_start = std::chrono::steady_clock::now();
  detail::ParallelTiles(partition.num_tiles(), options.outer_threads,
                        topology.affinity, options.schedule,
                        [&](uint64_t tile, unsigned) {
    if (!ok.load(std::memory_order_relaxed)) return;
    StateSpace state_space(1);
    auto allocation_start = std::chrono::steady_clock::now();
    auto state = options.numa
        ? state_space.CreateNuma(tile_q, topology.numa_nodes)
        : state_space.Create(tile_q);
    (void)allocation_start;
    if (state_space.IsNull(state)) { ok.store(false); return; }
    if (tile == 0) state_space.SetStateZero(state);
    else state_space.SetAllZeros(state);
    Simulator simulator(std::max(1U, options.inner_threads));
    typename QSimRunner<IO, Fuser, Factory>::Parameter param;
    param.max_fused_size = options.max_fused_size;
    param.seed = 1; param.verbosity = 0;
    if (!QSimRunner<IO, Fuser, Factory>::Run(
            param, tile_circuit, state_space, simulator, state)) {
      ok.store(false); return;
    }
    norms[tile] = state_space.Norm(state);
    if (tile == 0) {
      std::lock_guard<std::mutex> lock(first_mutex);
      const uint64_t count = std::min<uint64_t>(8, uint64_t{1} << tile_q);
      first.reserve(count);
      for (uint64_t i = 0; i < count; ++i) first.push_back(StateSpace::GetAmpl(state, i));
    }
  });
  if (!ok.load()) return false;
  if (stats) {
    stats->tiles = partition.num_tiles(); stats->batches = batches.size();
    stats->topology = topology;
    stats->backend = "simd-tile";
    stats->planning_seconds = std::chrono::duration<double>(planning_end - planning_start).count();
    stats->total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    stats->first_amplitudes.assign(first.begin(), first.end());
    for (auto norm : norms) stats->norm += norm;
  }
  return true;
}

// SIMD tile execution with persistent logical/physical remapping. High-bit
// swaps exchange tile ownership (or amplitudes between paired tiles), while
// low-bit swaps operate inside each backend state. Gates are then applied by
// the existing SIMD simulator with normalized physical qubit order.
template <typename IO, typename Fuser, typename Factory, typename Circuit>
bool RunQSimTiledSimdRemapped(const TiledOptions& options,
                              const Factory& factory, const Circuit& circuit,
                              TiledStats* stats = nullptr) {
  const auto topology = CpuTopology::Detect();
  const auto& ops = Operations<Circuit>::get(circuit);
  using Operation = typename std::decay_t<decltype(ops)>::value_type;
  using FP = OpFpType<Operation>;
  for (const auto& op : ops) {
    if (!OpGetAlternative<Measurement>(op) &&
        !OpGetAlternative<Gate<FP>>(op) &&
        !OpGetAlternative<ControlledGate<FP>>(op)) return false;
  }
  unsigned tile_q = options.tile_qubits;
  if (tile_q == 0) {
    tile_q = 1;
    while (tile_q < circuit.num_qubits &&
           (uint64_t{1} << (tile_q + 1)) * sizeof(FP) * 2 <= topology.TargetTileCacheBytes()) ++tile_q;
  }
  tile_q = std::min(tile_q, circuit.num_qubits);
  TilePartition partition(circuit.num_qubits, tile_q);
  auto planning_start = std::chrono::steady_clock::now();
  GateBatchPlanner<Operation> planner(ops, options.local_qubits
      ? options.local_qubits : tile_q, options.max_fused_size);
  auto batches = planner.Plan();
  auto planning_end = std::chrono::steady_clock::now();
  for (const auto& batch : batches) if (batch.qubits > tile_q) return false;

  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  auto total_start = std::chrono::steady_clock::now();
  StateSpace state_space(1);
  std::vector<State> states;
  states.reserve(partition.num_tiles());
  double allocation_seconds = 0;
  for (uint64_t tile = 0; tile < partition.num_tiles(); ++tile) {
    auto allocation_start = std::chrono::steady_clock::now();
    auto state = options.numa
        ? state_space.CreateNuma(tile_q, topology.numa_nodes)
        : state_space.Create(tile_q);
    allocation_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - allocation_start).count();
    if (state_space.IsNull(state)) return false;
    if (tile == 0) state_space.SetStateZero(state);
    else state_space.SetAllZeros(state);
    states.push_back(std::move(state));
  }

  QubitRemapper remapper(circuit.num_qubits);
  std::vector<QubitSwap> swaps;
  typename Fuser::Parameter param;
  param.max_fused_size = options.max_fused_size; param.verbosity = 0;
  double remap_seconds = 0, fusion_seconds = 0, gate_seconds = 0;
  double scheduling_seconds = 0;
  const unsigned workers = options.inner_threads > 1
      ? std::max(1U, options.outer_threads / options.inner_threads)
      : std::max(1U, options.outer_threads);
  std::mt19937 random(1);
  for (const auto& batch : batches) {
    auto remap_start = std::chrono::steady_clock::now();
    auto batch_swaps = remapper.MakeLocal(batch.logical_qubits, tile_q);
    for (const auto& swap : batch_swaps) {
      detail::ApplyTiledBitSwap<StateSpace>(states, tile_q,
                                             swap.physical_a, swap.physical_b,
                                             workers, topology.affinity,
                                             options.schedule);
      swaps.push_back(swap);
    }
    remap_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - remap_start).count();
    auto fusion_start = std::chrono::steady_clock::now();
    std::vector<Operation> batch_ops;
    batch_ops.reserve(batch.indices.size());
    for (auto index : batch.indices) batch_ops.push_back(ops[index]);
    auto fused = Fuser::template FuseGates<Operation>(
        param, circuit.num_qubits, batch_ops, true);
    fusion_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fusion_start).count();
    auto gate_start = std::chrono::steady_clock::now();
    for (const auto& operation : fused) {
      if (!detail::ApplySimdFused<FP, Operation, decltype(operation),
                                  Simulator, StateSpace>(
              operation, remapper.layout(), states, tile_q, workers,
              topology.affinity, options.schedule, random,
              options.inner_threads, &scheduling_seconds)) return false;
    }
    gate_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gate_start).count();
  }
  for (auto it = swaps.rbegin(); it != swaps.rend(); ++it)
    detail::ApplyTiledBitSwap<StateSpace>(states, tile_q,
                                          it->physical_a, it->physical_b,
                                          workers, topology.affinity,
                                          options.schedule);
  if (stats) {
    stats->tiles = partition.num_tiles(); stats->batches = batches.size();
    stats->topology = topology; stats->backend = "simd-remapped-tile";
    stats->planning_seconds = std::chrono::duration<double>(
        planning_end - planning_start).count();
    stats->allocation_seconds = allocation_seconds;
    stats->remap_seconds = remap_seconds; stats->fusion_seconds = fusion_seconds;
    stats->gate_seconds = gate_seconds;
    stats->scheduling_seconds = scheduling_seconds;
    stats->total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    const uint64_t count = std::min<uint64_t>(8, uint64_t{1} << circuit.num_qubits);
    stats->first_amplitudes.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t tile = i >> tile_q, local = i & ((uint64_t{1} << tile_q) - 1);
      stats->first_amplitudes.push_back(StateSpace::GetAmpl(states[tile], local));
    }
    for (const auto& state : states) stats->norm += state_space.Norm(state);
  }
  return true;
}

// Tiled execution entry point.  The state is allocated with a best-effort
// interleaved NUMA policy and the existing SIMD runner performs the actual
// gate kernels.  Keeping the execution call shared with qsim_base guarantees
// identical measurement and backend semantics while the tile scheduler and
// planner can be incrementally enabled by future kernels.
template <typename IO, typename Fuser, typename Factory, typename Circuit>
bool RunQSimTiled(const TiledOptions& options, const Factory& factory,
                  const Circuit& circuit, TiledStats* stats = nullptr) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  const auto topology = CpuTopology::Detect();
  unsigned tile_q = options.tile_qubits;
  if (tile_q == 0) {
    uint64_t target = 512 * 1024;
    target = topology.TargetTileCacheBytes();
    while (tile_q < circuit.num_qubits && (uint64_t{1} << (tile_q + 1)) * sizeof(typename StateSpace::fp_type) * 2 <= target) ++tile_q;
  }
  TilePartition partition(circuit.num_qubits, tile_q);
  TileScheduler scheduler(partition.num_tiles(), options.outer_threads, options.schedule);
  auto order = scheduler.StaticOrder();
  (void)order;
  const auto& ops = Operations<Circuit>::get(circuit);
  auto planning_start = std::chrono::steady_clock::now();
  GateBatchPlanner<typename std::decay_t<decltype(ops)>::value_type> planner(
      ops, options.local_qubits ? options.local_qubits : tile_q,
      options.max_fused_size);
  auto batches = planner.Plan();
  auto planning_end = std::chrono::steady_clock::now();

  StateSpace state_space = factory.CreateStateSpace();
  State state = options.numa ? state_space.CreateNuma(circuit.num_qubits, topology.numa_nodes)
                             : state_space.Create(circuit.num_qubits);
  if (state_space.IsNull(state)) return false;
  state_space.SetStateZero(state);
  typename QSimRunner<IO, Fuser, Factory>::Parameter param;
  param.max_fused_size = options.max_fused_size;
  param.seed = 1;
  param.verbosity = 0;
  bool ok = QSimRunner<IO, Fuser, Factory>::Run(param, circuit, state_space,
                                                 factory.CreateSimulator(), state);
  if (stats) {
    stats->tiles = partition.num_tiles(); stats->batches = batches.size(); stats->topology = topology;
    stats->backend = "simd-fallback";
    stats->planning_seconds = std::chrono::duration<double>(planning_end - planning_start).count();
    const uint64_t count = std::min<uint64_t>(8, uint64_t{1} << circuit.num_qubits);
    stats->first_amplitudes.reserve(count);
    for (uint64_t i = 0; i < count; ++i)
      stats->first_amplitudes.push_back(StateSpace::GetAmpl(state, i));
    stats->norm = state_space.Norm(state);
  }
  return ok;
}
}
#endif
