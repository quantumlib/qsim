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

// Gate-batched cache-local simulation runner.
//
// The state vector is split into tiles of 2^tile_qubits amplitudes, small
// enough to stay in cache. A gate can only run tile by tile when all of its
// qubits sit at low physical bit positions (below tile_qubits), so the runner
// repeatedly picks a batch of gates, swaps their qubits down into the tile,
// and runs the whole batch over each tile in one pass over memory.
//
// Each logical qubit lives at a physical bit position of the amplitude index
// (see QubitLayout). Positions are divided into four zones:
//
// 0           lane_qubits      eviction_floor     tile_qubits       num_qubits
// | lane zone |    hot zone    |    remap zone    | tile index bits |
//
//   lane zone   SIMD lanes; never remapped.
//   hot zone    Filled once, before the first batch, with the most-used
//               qubits; never evicted afterwards.
//   remap zone  Remap slots. Before each batch, qubits the batch needs that sit
//               above tile_qubits are swapped into slots held by qubits the
//               batch does not need.
//   tile index  Select which tile an amplitude belongs to; gates cannot act on
//               qubits stored here.
//
// Planning (GateBatchPlanner) walks the pending gates in circuit order and
// admits each gate whose qubits still fit into the remap slots and that may
// run before every gate deferred so far. It tries several starting sets of
// remap slots ("seeds") and keeps the batch with the best score: gates
// admitted minus an estimated swap cost.
//
// Reading guide:
//
//   GateBatchRunner::Run
//     SimulateCircuit
//       BuildPendingGates          wrap each gate, flag diagonal ones
//       SimulateSingleTile         whole state fits one tile:
//         FuseBatchGates, ExecuteFusedGates  fuse the circuit, run it once
//       SimulateMultiTile
//         FillHotZone              move the most-used qubits there, once
//         loop ExecuteNextGateBatch
//           GateBatchPlanner::PlanNextGateBatch
//           MoveQubitsIntoZone     swap batch qubits into the remap zone
//           BuildBatchOperations   rewrite gates to physical qubits
//           FuseBatchGates, ExecuteFusedGates
//         RestoreIdentityQubitOrder  only for Run(State&); the
//                                    QubitMappedState overload keeps the
//                                    permuted layout

#ifndef GATE_BATCH_RUNNER_H_
#define GATE_BATCH_RUNNER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "gate.h"
#include "gate_batch_planner.h"
#include "matrix.h"
#include "operation_base.h"
#include "qubit_layout.h"
#include "util.h"

namespace qsim {

namespace gate_batch_internal {

template <typename FP>
struct ExecutableGate {
  Qubits qubits;
  Matrix<FP> matrix;
};

}  // namespace gate_batch_internal

template <typename IO, typename Fuser, typename Backend>
class GateBatchRunner {
 public:
  using StateSpace = typename Backend::StateSpace;
  using State = typename Backend::State;
  using QubitMappedState = qsim::QubitMappedState<State>;
  using fp_type = typename Backend::fp_type;
  static_assert(
      std::is_same_v<fp_type, float>,
      "GateBatchRunner requires a float state space.");

  struct Parameter : public Fuser::Parameter, public Backend::Parameter {
    Parameter() { this->max_fused_size = 3; }

    // Unset lets the backend choose; a set value is used as given.
    std::optional<unsigned> tile_qubits;
    unsigned min_eviction_floor = Backend::kDefaultEvictionFloor;
    // Most lookahead seeds the planner tries per batch; see PlanNextGateBatch.
    unsigned max_lookahead_seeds = 64;
  };

  // Leaves `state` in standard qubit order, like QSimRunner::Run.
  template <typename Circuit>
  static bool Run(
      const Parameter& param, const Circuit& circuit, State& state) {
    QubitLayout layout(circuit.num_qubits);
    return RunWithLayout(
        param, circuit, state, layout, QubitOrderOnExit::kRestoreIdentity);
  }

  // Leaves `state` permuted, with `state.layout` describing the order; this
  // skips the restore passes.
  template <typename Circuit>
  static bool Run(
      const Parameter& param, const Circuit& circuit, QubitMappedState& state) {
    return RunWithLayout(
        param, circuit, state.state, state.layout,
        QubitOrderOnExit::kKeepPermuted);
  }

 private:
  using PendingGateList = gate_batch_internal::PendingGateList<fp_type>;
  using ExecutableGate = gate_batch_internal::ExecutableGate<fp_type>;
  using GateBatchPlanner = gate_batch_internal::GateBatchPlanner<fp_type>;
  using GateBatchPlan = gate_batch_internal::GateBatchPlan;
  using QubitZones = gate_batch_internal::QubitZones;

  // Counters and timings reported by LogSimulationSummary.
  struct SimulationStats {
    unsigned num_gate_batches = 0;
    unsigned num_restore_passes = 0;
    unsigned num_raw_gates = 0;
    unsigned num_swaps = 0;
    unsigned num_executable_gates = 0;
    double prepare_seconds = 0.0;
    double plan_seconds = 0.0;
    double swap_seconds = 0.0;
    double fuse_seconds = 0.0;
    double gate_seconds = 0.0;
  };

  // Whether Run undoes the qubit permutation the batches leave behind.
  enum class QubitOrderOnExit { kRestoreIdentity, kKeepPermuted };

  template <typename Circuit>
  static bool RunWithLayout(
      const Parameter& param, const Circuit& circuit, State& state,
      QubitLayout& layout, QubitOrderOnExit order) {
    if (!SizesMatch(circuit.num_qubits, state.num_qubits()) ||
        !SizesMatch(circuit.num_qubits, layout.NumQubits())) {
      return false;
    }
    GateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuit(circuit, order);
  }

  GateBatchRunner(
      const Parameter& param, unsigned num_qubits, State& state,
      QubitLayout& layout)
      : param_(param),
        backend_(param, num_qubits, state),
        zones_(
            QubitZones::ForBackend(
                backend_, num_qubits, param.min_eviction_floor)),
        layout_(layout) {
    assert(layout_.NumQubits() == num_qubits);
  }

  static bool SizesMatch(unsigned circuit_qubits, unsigned state_qubits) {
    if (circuit_qubits == state_qubits) return true;
    IO::errorf(
        "qsim_gate_batch: the circuit has %u qubits but the state "
        "has %u.\n",
        circuit_qubits, state_qubits);
    return false;
  }

  template <typename Circuit>
  bool SimulateCircuit(const Circuit& circuit, QubitOrderOnExit order) {
    if (!ValidateParameters() || !backend_.Prepare()) return false;
    simulation_stats_.num_raw_gates = unsigned(circuit.ops.size());

    auto pending_gates = TimeHostWork(simulation_stats_.prepare_seconds, [&] {
      return BuildPendingGates(circuit);
    });
    if (!pending_gates) return false;

    using Op = typename std::decay_t<decltype(circuit.ops)>::value_type;
    if (zones_.IsSingleTile()) return SimulateSingleTile<Op>(*pending_gates);
    return SimulateMultiTile<Op>(*pending_gates, order);
  }

  bool ValidateParameters() const {
    if (param_.max_fused_size < 2) {
      IO::errorf("qsim_gate_batch: max_fused_size must be at least 2.\n");
      return false;
    }
    if (param_.max_fused_size > Backend::kMaxGateQubits) {
      IO::errorf(
          "qsim_gate_batch: max_fused_size must be at most %u.\n",
          Backend::kMaxGateQubits);
      return false;
    }
    return true;
  }

  // The whole state is one tile: no planning or swaps. Every gate runs in one
  // batch, on the physical positions the layout gives its qubits. Gates keep
  // their circuit times, so the fuser groups them as for the whole circuit.
  template <typename Op>
  bool SimulateSingleTile(const PendingGateList& pending_gates) {
    std::vector<Op> operations;
    operations.reserve(pending_gates.Size());
    for (const auto* gate = pending_gates.NextPending(); gate != nullptr;
         gate = pending_gates.NextPending(gate)) {
      operations.push_back(
          Op{MakePhysicalGate(*gate->gate, layout_, gate->gate->time)});
    }

    const auto executable_gates =
        TimeHostWork(simulation_stats_.fuse_seconds, [&] {
          return FuseBatchGates(operations);
        });
    if (!executable_gates) return false;

    const auto simulation_start = StartSimulationClock();
    ExecuteFusedGates(*executable_gates);
    simulation_stats_.num_gate_batches = operations.empty() ? 0 : 1;
    FinishSimulation(simulation_start);
    return true;
  }

  template <typename Op>
  bool SimulateMultiTile(
      PendingGateList& pending_gates, QubitOrderOnExit order) {
    const auto simulation_start = StartSimulationClock();
    FillHotZone(pending_gates);

    const GateBatchPlanner planner(
        zones_, pending_gates, layout_, param_.max_lookahead_seeds);
    while (pending_gates.NextPending() != nullptr) {
      if (!ExecuteNextGateBatch<Op>(planner, pending_gates)) return false;
    }

    if (order == QubitOrderOnExit::kRestoreIdentity) {
      RestoreIdentityQubitOrder();
    }
    FinishSimulation(simulation_start);
    return true;
  }

  // Returns false after reporting an error (a gate that never fits, or a
  // fuser failure).
  template <typename Op>
  bool ExecuteNextGateBatch(
      const GateBatchPlanner& planner, PendingGateList& pending_gates) {
    const auto plan = TimeHostWork(simulation_stats_.plan_seconds, [&] {
      return planner.PlanNextGateBatch();
    });
    if (!plan.HasGates()) {
      IO::errorf(
          "qsim_gate_batch: a gate does not fit the remap slots "
          "of %u tile qubits; use a larger tile_qubits.\n",
          zones_.tile_qubits);
      return false;
    }

    MoveQubitsIntoZone(
        plan.uses_qubit, zones_.eviction_floor, zones_.tile_qubits);

    const auto executable_gates =
        TimeHostWork(simulation_stats_.fuse_seconds, [&] {
          const auto batch_operations =
              BuildBatchOperations<Op>(plan, pending_gates);
          return FuseBatchGates(batch_operations);
        });
    if (!executable_gates) return false;
    pending_gates.MarkApplied(plan.gate_indices);

    ExecuteFusedGates(*executable_gates);
    ++simulation_stats_.num_gate_batches;
    return true;
  }

  template <typename Circuit>
  static std::optional<PendingGateList> BuildPendingGates(
      const Circuit& circuit) {
    auto pending_gates =
        PendingGateList::FromCircuit(circuit, Backend::kMaxGateQubits);
    if (!pending_gates) {
      IO::errorf(
          "qsim_gate_batch: only gates of up to %u qubits (counting controls) "
          "are supported, not measurements or other operations; use "
          "QSimRunner for those.\n",
          Backend::kMaxGateQubits);
    }
    return pending_gates;
  }

  std::vector<uint64_t> ComputeQubitUsageScores(
      const PendingGateList& pending_gates) const {
    std::vector<uint64_t> scores(zones_.num_qubits, 0);
    for (std::size_t idx = 0; idx < pending_gates.Size(); ++idx) {
      const auto& qubits = pending_gates[idx].Qubits();
      for (unsigned q : qubits) scores[q] += qubits.size();
    }
    return scores;
  }

  static std::vector<char> SelectHighestScoringQubits(
      const std::vector<uint64_t>& scores, Qubits candidates, unsigned count) {
    assert(count <= candidates.size());
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&scores](unsigned a, unsigned b) { return scores[a] > scores[b]; });

    std::vector<char> selected(scores.size(), 0);
    for (unsigned i = 0; i < count; ++i) selected[candidates[i]] = 1;
    return selected;
  }

  void FillHotZone(const PendingGateList& pending_gates) {
    if (zones_.NumHotSlots() == 0) return;

    // Lane qubits never move, so only qubits stored above the lanes can be
    // made hot.
    Qubits candidates;
    for (unsigned q = 0; q < zones_.num_qubits; ++q) {
      if (layout_.PhysicalPositionOf(q) >= zones_.lane_qubits) {
        candidates.push_back(q);
      }
    }

    const auto usage_scores = ComputeQubitUsageScores(pending_gates);
    const auto is_hot = SelectHighestScoringQubits(
        usage_scores, std::move(candidates), zones_.NumHotSlots());

    const auto num_swaps =
        MoveQubitsIntoZone(is_hot, zones_.lane_qubits, zones_.eviction_floor);
    LogHotZonePlacement(usage_scores, num_swaps);
  }

  // Swaps each wanted qubit stored at or above zone_end into a position in
  // [zone_begin, zone_end) held by an unwanted qubit, taking positions from the
  // top of the zone down. Updates both the layout and the state, and returns
  // the number of swaps.
  //
  // The zone always has enough unwanted positions, so eviction never reaches
  // below zone_begin: the hot zone holds exactly as many positions as there
  // are hot qubits, and a batch plan's wanted qubits outside the hot and lane
  // zones never outnumber the remap slots.
  std::size_t MoveQubitsIntoZone(
      const std::vector<char>& wanted, unsigned zone_begin, unsigned zone_end) {
    std::vector<PositionSwap> swaps;
    auto evict_position = zone_end;

    for (unsigned q = 0; q < wanted.size(); ++q) {
      if (!wanted[q]) continue;
      const auto position = layout_.PhysicalPositionOf(q);
      if (position < zone_end) continue;

      do {
        assert(evict_position > zone_begin);
        --evict_position;
      } while (wanted[layout_.LogicalQubitAt(evict_position)]);
      swaps.push_back({evict_position, position});
      layout_.SwapPhysicalPositions(evict_position, position);
    }

    ApplySwapsToState(swaps);
    return swaps.size();
  }

  void ApplySwapsToState(const std::vector<PositionSwap>& swaps) {
    if (swaps.empty()) return;
    TimeBackendWork(simulation_stats_.swap_seconds, [&] {
      backend_.ApplySwaps(swaps);
    });
    simulation_stats_.num_swaps += swaps.size();
  }

  void RestoreIdentityQubitOrder() {
    while (true) {
      const auto swaps = layout_.StepTowardIdentity();
      if (swaps.empty()) break;
      ApplySwapsToState(swaps);
      ++simulation_stats_.num_restore_passes;
    }
  }

  template <typename Op>
  std::vector<Op> BuildBatchOperations(
      const GateBatchPlan& plan, const PendingGateList& pending_gates) const {
    std::vector<Op> batch_operations;
    batch_operations.reserve(plan.NumGates());

    unsigned time = 0;
    for (std::size_t idx : plan.gate_indices) {
      batch_operations.push_back(
          Op{MakePhysicalGate(*pending_gates[idx].gate, layout_, time++)});
    }
    return batch_operations;
  }

  // Returns `raw_gate` rewritten to act on the physical positions that `layout`
  // stores its logical qubits at, with the qubits sorted as the simulator
  // expects.
  static Gate<fp_type> MakePhysicalGate(
      const Gate<fp_type>& raw_gate, const QubitLayout& layout, unsigned time) {
    // kind, params and swapped keep their zero defaults; the fuser ignores
    // them.
    Gate<fp_type> physical_gate{};
    physical_gate.time = time;
    for (unsigned q : raw_gate.qubits) {
      physical_gate.qubits.push_back(layout.PhysicalPositionOf(q));
    }
    physical_gate.matrix = raw_gate.matrix;
    NormalizeGateQubitOrder(physical_gate.qubits, physical_gate.matrix);
    return physical_gate;
  }

  // Sorts the qubits in ascending order, as the simulator expects, and permutes
  // the matrix to match.
  static void NormalizeGateQubitOrder(
      Qubits& qubits, Matrix<fp_type>& matrix) {
    if (qubits.size() < 2) return;
    auto permutation = NormalToGateOrderPermutation(qubits);
    if (!permutation.empty()) {
      MatrixShuffle(permutation, qubits.size(), matrix);
      std::sort(qubits.begin(), qubits.end());
    }
  }

  template <typename Op>
  std::optional<std::vector<ExecutableGate>> FuseBatchGates(
      const std::vector<Op>& batch_operations) {
    auto fused_ops =
        Fuser::FuseGates(param_, zones_.tile_qubits, batch_operations);
    if (fused_ops.empty() && !batch_operations.empty()) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return std::nullopt;
    }

    std::vector<ExecutableGate> executable_gates;
    executable_gates.reserve(fused_ops.size());
    for (auto& operation : fused_ops) {
      // Only gates reach the fuser, each at most kMaxGateQubits wide.
      auto* fused = OpGetAlternative<FusedGate<fp_type>>(operation);
      assert(
          fused != nullptr && fused->qubits.size() <= Backend::kMaxGateQubits);
      assert(!fused->matrix.empty());
      assert(std::is_sorted(fused->qubits.begin(), fused->qubits.end()));
      executable_gates.push_back(
          {std::move(fused->qubits), std::move(fused->matrix)});
    }
    simulation_stats_.num_executable_gates += executable_gates.size();
    return executable_gates;
  }

  void ExecuteFusedGates(const std::vector<ExecutableGate>& executable_gates) {
    TimeBackendWork(simulation_stats_.gate_seconds, [&] {
      backend_.ExecuteBatch(executable_gates);
    });
  }

  double StartSimulationClock() {
    backend_.Synchronize();
    return GetTime();
  }

  // Waits for the backend to finish all work, then logs the run's summary.
  void FinishSimulation(double simulation_start) {
    backend_.Synchronize();
    LogSimulationSummary(simulation_start);
  }

  // Adds the wall time of host-side work (preparation, planning, fusion) to
  // `seconds` and returns the work's result.
  template <typename Work>
  auto TimeHostWork(double& seconds, Work&& work) {
    const auto start = GetTime();
    auto result = work();
    seconds += GetTime() - start;
    return result;
  }

  // Adds the time of backend work (gate and swap passes) to `seconds`. Backend
  // calls may return before the work finishes, so at verbosity > 1 this
  // synchronizes before and after to time the work itself; otherwise the time
  // covers only the call.
  template <typename Work>
  void TimeBackendWork(double& seconds, Work&& work) {
    if (param_.verbosity > 1) backend_.Synchronize();
    const auto start = GetTime();
    work();
    if (param_.verbosity > 1) backend_.Synchronize();
    seconds += GetTime() - start;
  }

  void LogHotZonePlacement(
      const std::vector<uint64_t>& usage_scores,
      std::size_t num_initial_swaps) const {
    if (param_.verbosity <= 1) return;

    IO::messagef(
        "hot zone [%u,%u):", zones_.lane_qubits, zones_.eviction_floor);
    for (unsigned p = zones_.lane_qubits; p < zones_.eviction_floor; ++p) {
      const auto q = layout_.LogicalQubitAt(p);
      IO::messagef(
          " q%u(%llu)", q, static_cast<unsigned long long>(usage_scores[q]));
    }
    IO::messagef("; %u initial swaps.\n", unsigned(num_initial_swaps));
  }

  void LogSimulationSummary(double simulation_start) const {
    if (param_.verbosity == 0) return;

    IO::messagef("simu time is %g seconds.\n", GetTime() - simulation_start);
    IO::messagef(
        "gate-batch runner: %u gate batches, %u restore passes, "
        "%u raw gates, %u executable gates, "
        "%u qubit swaps.\n",
        simulation_stats_.num_gate_batches,
        simulation_stats_.num_restore_passes, simulation_stats_.num_raw_gates,
        simulation_stats_.num_executable_gates, simulation_stats_.num_swaps);
    if (param_.verbosity > 1) {
      IO::messagef(
          "breakdown: prepare %g s, plan %g s, swap passes %g s, fuse %g s, "
          "gate passes %g s.\n",
          simulation_stats_.prepare_seconds, simulation_stats_.plan_seconds,
          simulation_stats_.swap_seconds, simulation_stats_.fuse_seconds,
          simulation_stats_.gate_seconds);
    }
  }

  const Parameter& param_;
  Backend backend_;
  const QubitZones zones_;
  QubitLayout& layout_;

  // Accumulated over the whole run; printed by LogSimulationSummary.
  SimulationStats simulation_stats_;
};

}  // namespace qsim

#endif  // GATE_BATCH_RUNNER_H_
