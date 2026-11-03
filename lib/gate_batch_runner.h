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
// targets sit at low physical bit positions (below tile_qubits), so the runner
// repeatedly picks a batch of gates, swaps their targets down into the tile,
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
//   tile index  Select which tile an amplitude belongs to; controls may stay
//               here, but targets must be brought into the tile.
//
// Planning (GateBatchPlanner) walks the pending gates in circuit order and
// admits each gate whose targets fit into the remap slots and that may
// run before every gate deferred so far. It tries several starting sets of
// remap slots ("seeds") and keeps the batch with the best score: gates
// admitted minus an estimated swap cost.
//
// Reading guide:
//
//   GateBatchRunner::Run
//     SimulateCircuit / SimulateCircuitWithCallback
//       PrepareSimulation          validate parameters, extract gates/boundaries
//       FillHotZone                place frequently used targets, once
//       ExecuteThroughTimeStep     drain measurements and gates up to a window
//       ExecuteRemainingOperations drain remaining measurement boundaries/gates
//         ExecuteGatesUntil        drain gates before a boundary
//           SimulateSingleTile     one batch when the state fits one tile
//           ExecuteNextGateBatch   otherwise plan, remap, and execute batches
//         Measure                  sample and collapse the full state
//       RestoreIdentityQubitOrder  for State and callback overloads
//       FinishSimulation

#ifndef GATE_BATCH_RUNNER_H_
#define GATE_BATCH_RUNNER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <variant>
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
  Qubits controls;                 // physical positions inside the tile
  uint64_t control_values = 0;     // bit i belongs to controls[i]
  uint64_t tile_control_mask = 0;  // positions relative to the tile index
  uint64_t tile_control_values = 0;
};

template <typename T, typename = void>
struct HasMaxFusedSize : std::false_type {};

template <typename T>
struct HasMaxFusedSize<
    T, std::void_t<decltype(std::declval<T>().max_fused_size)>>
    : std::true_type {};

}  // namespace gate_batch_internal

template <typename IO, typename Fuser, typename Backend,
          typename RGen = std::mt19937>
class GateBatchRunner {
 public:
  using StateSpace = typename Backend::StateSpace;
  using State = typename Backend::State;
  using MeasurementResult = typename StateSpace::MeasurementResult;
  using QubitMappedState = qsim::QubitMappedState<State>;
  using fp_type = typename Backend::fp_type;
  static_assert(
      std::is_same_v<fp_type, float>,
      "GateBatchRunner requires a float state space.");

  template <typename MeasurementFunc>
  using EnableIfMeasurementFunc = std::enable_if_t<
      std::is_invocable_v<
          MeasurementFunc, unsigned, const StateSpace&, const State&>,
      int>;

  template <typename StateOrMappedState>
  using EnableIfState = std::enable_if_t<
      std::is_same_v<StateOrMappedState, State> ||
          std::is_same_v<StateOrMappedState, QubitMappedState>,
      int>;

  struct Parameter : public Fuser::Parameter, public Backend::Parameter {
    Parameter() {
      if constexpr (gate_batch_internal::HasMaxFusedSize<
                        typename Fuser::Parameter>::value) {
        this->max_fused_size = 3;
      }
    }

    // Unset lets the backend choose; a set value is used as given.
    std::optional<unsigned> tile_qubits;
    uint64_t seed = 1;
    unsigned min_eviction_floor = Backend::kDefaultEvictionFloor;
    // Most lookahead seeds the planner tries per batch; see PlanNextGateBatch.
    unsigned max_lookahead_seeds = 64;
  };

  // Leaves `state` in standard qubit order.
  template <typename Circuit>
  static bool Run(
      const Parameter& param, const Circuit& circuit, State& state) {
    std::vector<MeasurementResult> discarded_results;
    return Run(param, circuit, state, discarded_results);
  }

  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit, State& state,
                  std::vector<MeasurementResult>& measurement_results) {
    QubitLayout layout(circuit.num_qubits);
    return RunWithLayout(param, circuit, state, layout, measurement_results,
                         QubitOrderOnExit::kRestoreIdentity);
  }

  // Leaves `state` permuted, with `state.layout` describing the order.
  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  QubitMappedState& state) {
    std::vector<MeasurementResult> discarded_results;
    return Run(param, circuit, state, discarded_results);
  }

  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  QubitMappedState& state,
                  std::vector<MeasurementResult>& measurement_results) {
    return RunWithLayout(param, circuit, state.state, state.layout,
                         measurement_results, QubitOrderOnExit::kKeepPermuted);
  }

  // Allocates a zero-initialized state and invokes `measure(k, state_space,
  // state)` after all circuit operations at or before the final timestep (or
  // each timestep in `times_to_measure_at`), with `state` in standard qubit
  // order.
  template <typename Circuit, typename MeasurementFunc,
            EnableIfMeasurementFunc<MeasurementFunc> = 0>
  static bool Run(
      const Parameter& param, const Circuit& circuit,
      MeasurementFunc measure) {
    if (circuit.ops.empty()) {
      return Run(param, std::vector<unsigned>{}, circuit, std::move(measure));
    }
    return Run(
        param, {OpTime(circuit.ops.back())}, circuit, std::move(measure));
  }

  template <typename Circuit, typename MeasurementFunc,
            EnableIfMeasurementFunc<MeasurementFunc> = 0>
  static bool Run(
      const Parameter& param,
      const std::vector<unsigned>& times_to_measure_at,
      const Circuit& circuit, MeasurementFunc measure) {
    StateSpace state_space = Backend::CreateStateSpace(param);
    State state = state_space.Create(circuit.num_qubits);
    if (state_space.IsNull(state)) {
      IO::errorf("not enough memory: is the number of qubits too large?\n");
      return false;
    }
    state_space.SetStateZero(state);
    QubitLayout layout(circuit.num_qubits);
    GateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuitWithCallback(
        circuit, times_to_measure_at, state_space, state, measure);
  }

 private:
  // Gate qubit indices have been mapped through QubitLayout to physical bit
  // positions.
  using MappedGate = std::variant<Gate<fp_type>, ControlledGate<fp_type>>;
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
    unsigned num_measurements = 0;
    double prepare_seconds = 0.0;
    double plan_seconds = 0.0;
    double swap_seconds = 0.0;
    double fuse_seconds = 0.0;
    double gate_seconds = 0.0;
    double measure_seconds = 0.0;
  };

  // Whether Run undoes the qubit permutation the batches leave behind.
  enum class QubitOrderOnExit { kRestoreIdentity, kKeepPermuted };

  template <typename Circuit>
  static bool RunWithLayout(
      const Parameter& param, const Circuit& circuit, State& state,
      QubitLayout& layout, std::vector<MeasurementResult>& measurement_results,
      QubitOrderOnExit order) {
    if (!SizesMatch(circuit.num_qubits, state.num_qubits()) ||
        !SizesMatch(circuit.num_qubits, layout.NumQubits())) {
      return false;
    }
    GateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuit(circuit, order, measurement_results);
  }

  GateBatchRunner(
      const Parameter& param, unsigned num_qubits, State& state,
      QubitLayout& layout)
      : param_(param),
        backend_(param, num_qubits, state),
        zones_(
            QubitZones::ForBackend(
                backend_, num_qubits, param.min_eviction_floor)),
        layout_(layout),
        rgen_(param.seed) {
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
  bool SimulateCircuit(const Circuit& circuit, QubitOrderOnExit order,
                       std::vector<MeasurementResult>& measurement_results) {
    auto pending_gates = PrepareSimulation(circuit);
    if (!pending_gates) return false;

    const auto simulation_start = StartSimulationClock();
    if (!zones_.IsSingleTile() && pending_gates->Size() != 0) {
      FillHotZone(*pending_gates);
    }
    std::size_t boundary_index = 0;
    if (!ExecuteRemainingOperations(
            *pending_gates, boundary_index, measurement_results)) {
      return false;
    }
    if (order == QubitOrderOnExit::kRestoreIdentity) {
      RestoreIdentityQubitOrder();
    }
    FinishSimulation(simulation_start);
    return true;
  }

  template <typename Circuit, typename MeasurementFunc>
  bool SimulateCircuitWithCallback(
      const Circuit& circuit,
      const std::vector<unsigned>& times_to_measure_at,
      const StateSpace& state_space, State& state, MeasurementFunc& measure) {
    auto pending_gates = PrepareSimulation(circuit, times_to_measure_at);
    if (!pending_gates) return false;

    const auto simulation_start = StartSimulationClock();
    if (!zones_.IsSingleTile() && pending_gates->Size() != 0) {
      FillHotZone(*pending_gates);
    }

    if (!circuit.ops.empty()) {
      std::vector<MeasurementResult> discarded_results;
      std::size_t boundary_index = 0;
      for (std::size_t k = 0; k < times_to_measure_at.size(); ++k) {
        const unsigned time_step = times_to_measure_at[k];
        const auto gate_end = pending_gates->GateEndAtOrBefore(time_step);
        if (!ExecuteThroughTimeStep(
                *pending_gates, boundary_index, time_step, gate_end,
                discarded_results)) {
          return false;
        }
        RestoreIdentityQubitOrder();
        measure(unsigned(k), state_space, state);
        if (boundary_index == pending_gates->MeasurementBoundaries().size() &&
            gate_end == pending_gates->Size()) {
          break;
        }
      }
      if (!ExecuteRemainingOperations(
              *pending_gates, boundary_index, discarded_results)) {
        return false;
      }
    }
    FinishSimulation(simulation_start);
    return true;
  }

  template <typename Circuit>
  std::optional<PendingGateList> PrepareSimulation(
      const Circuit& circuit,
      const std::vector<unsigned>& times_to_measure_at = {}) {
    if (!ValidateParameters() || !backend_.Prepare()) return std::nullopt;

    auto pending_gates = TimeHostWork(simulation_stats_.prepare_seconds, [&] {
      return BuildPendingGates(circuit, times_to_measure_at);
    });
    if (!pending_gates) return std::nullopt;
    simulation_stats_.num_raw_gates = unsigned(pending_gates->Size());
    zones_ = QubitZones::ForBackend(
        backend_, zones_.num_qubits, param_.min_eviction_floor,
        pending_gates->MaxTargetQubits());
    return pending_gates;
  }

  bool ExecuteThroughTimeStep(
      PendingGateList& pending_gates, std::size_t& boundary_index,
      unsigned time_step, std::size_t gate_end,
      std::vector<MeasurementResult>& measurement_results) {
    const auto& boundaries = pending_gates.MeasurementBoundaries();
    while (boundary_index < boundaries.size() &&
           boundaries[boundary_index].measurement.time <= time_step) {
      if (!ExecuteGatesUntil(
              pending_gates, boundaries[boundary_index].gate_end) ||
          !Measure(boundaries[boundary_index].measurement,
                   measurement_results)) {
        return false;
      }
      ++boundary_index;
    }
    return ExecuteGatesUntil(pending_gates, gate_end);
  }

  bool ExecuteRemainingOperations(
      PendingGateList& pending_gates, std::size_t& boundary_index,
      std::vector<MeasurementResult>& measurement_results) {
    const auto& boundaries = pending_gates.MeasurementBoundaries();
    while (boundary_index < boundaries.size()) {
      if (!ExecuteGatesUntil(
              pending_gates, boundaries[boundary_index].gate_end) ||
          !Measure(boundaries[boundary_index].measurement,
                   measurement_results)) {
        return false;
      }
      ++boundary_index;
    }
    return ExecuteGatesUntil(pending_gates, pending_gates.Size());
  }

  bool ValidateParameters() const {
    if constexpr (gate_batch_internal::HasMaxFusedSize<
                      typename Fuser::Parameter>::value) {
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
    }
    return true;
  }

  // Execute the pending gates before the boundary in one tile. Circuit
  // times are retained for fusion dependency tracking.
  bool SimulateSingleTile(PendingGateList& pending_gates) {
    std::vector<MappedGate> mapped_gates;
    std::vector<std::size_t> indices;
    mapped_gates.reserve(pending_gates.Size());
    for (const auto* gate = pending_gates.NextPending(); gate != nullptr;
         gate = pending_gates.NextPending(gate)) {
      mapped_gates.push_back(MakeMappedGate(*gate, gate->gate->time));
      indices.push_back(gate->index);
    }

    const auto executable_gates =
        TimeHostWork(simulation_stats_.fuse_seconds, [&] {
          return BuildExecutableGates(mapped_gates);
        });
    if (!executable_gates) return false;

    ExecuteFusedGates(*executable_gates);
    pending_gates.MarkApplied(indices);
    simulation_stats_.num_gate_batches += mapped_gates.empty() ? 0 : 1;
    return true;
  }

  bool ExecuteGatesUntil(PendingGateList& pending_gates, std::size_t gate_end) {
    pending_gates.SetGateEnd(gate_end);
    return ExecutePendingGates(pending_gates);
  }

  bool ExecutePendingGates(PendingGateList& pending_gates) {
    if (pending_gates.NextPending() == nullptr) return true;
    if (zones_.IsSingleTile()) return SimulateSingleTile(pending_gates);

    const GateBatchPlanner planner(
        zones_, pending_gates, layout_, param_.max_lookahead_seeds);
    while (pending_gates.NextPending() != nullptr) {
      if (!ExecuteNextGateBatch(planner, pending_gates)) return false;
    }
    return true;
  }

  bool Measure(const Measurement& measurement,
               std::vector<MeasurementResult>& measurement_results) {
    Qubits physical_qubits;
    for (unsigned q : measurement.qubits) {
      physical_qubits.push_back(layout_.PhysicalPositionOf(q));
    }
    auto result = TimeBackendWork(simulation_stats_.measure_seconds, [&] {
      return backend_.Measure(physical_qubits, rgen_);
    });
    if (!result.valid) {
      IO::errorf("qsim_gate_batch: measurement failed.\n");
      return false;
    }
    // bitstring follows the requested qubit order. Rewrite position-indexed
    // fields into logical coordinates before exposing the result.
    result.mask = 0;
    result.bits = 0;
    for (std::size_t i = 0; i < measurement.qubits.size(); ++i) {
      const auto bit = uint64_t{1} << measurement.qubits[i];
      result.mask |= bit;
      if (result.bitstring[i]) result.bits |= bit;
    }
    measurement_results.push_back(std::move(result));
    ++simulation_stats_.num_measurements;
    return true;
  }

  // Returns false after reporting an error (a gate that never fits, or a
  // fuser failure).
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
          const auto batch_gates =
              BuildBatchGates(plan, pending_gates);
          return BuildExecutableGates(batch_gates);
        });
    if (!executable_gates) return false;
    pending_gates.MarkApplied(plan.gate_indices);

    ExecuteFusedGates(*executable_gates);
    ++simulation_stats_.num_gate_batches;
    return true;
  }

  template <typename Circuit>
  static std::optional<PendingGateList> BuildPendingGates(
      const Circuit& circuit,
      const std::vector<unsigned>& times_to_measure_at = {}) {
    auto pending_gates =
        PendingGateList::FromCircuit(circuit, Backend::kMaxGateQubits,
                                     Backend::kMaxControlledGateQubits,
                                     times_to_measure_at);
    if (!pending_gates) {
      IO::errorf(
          "qsim_gate_batch: supports ordinary gates up to %u targets and "
          "controlled gates up to %u targets, and measurements with valid "
          "time boundaries and qubits. Other operations are unsupported.\n",
          Backend::kMaxGateQubits, Backend::kMaxControlledGateQubits);
    }
    return pending_gates;
  }

  std::vector<uint64_t> ComputeQubitUsageScores(
      const PendingGateList& pending_gates) const {
    std::vector<uint64_t> scores(zones_.num_qubits, 0);
    for (std::size_t idx = 0; idx < pending_gates.Size(); ++idx) {
      const auto& qubits = pending_gates[idx].TargetQubits();
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

  std::vector<MappedGate> BuildBatchGates(
      const GateBatchPlan& plan, const PendingGateList& pending_gates) const {
    std::vector<MappedGate> batch_gates;
    batch_gates.reserve(plan.NumGates());

    // Gates sharing any target or control retain their order. Independent
    // gates share layers so the fuser can combine them.
    std::vector<unsigned> next_time(zones_.num_qubits, 0);
    for (std::size_t idx : plan.gate_indices) {
      const auto& gate = pending_gates[idx];
      unsigned time = 0;
      for (unsigned q : gate.Qubits()) time = std::max(time, next_time[q]);
      for (unsigned q : gate.Qubits()) next_time[q] = time + 1;
      batch_gates.push_back(MakeMappedGate(gate, time));
    }
    std::stable_sort(batch_gates.begin(), batch_gates.end(),
                     [](const MappedGate& a, const MappedGate& b) {
                       return OpTime(a) < OpTime(b);
                     });
    return batch_gates;
  }

  // Target matrix order and control-value order are independent. Sorting the
  // mapped controls must move each required value with its control qubit.
  MappedGate MakeMappedGate(
      const gate_batch_internal::PendingGate<fp_type>& pending,
      unsigned time) const {
    auto target = MakePhysicalGate(*pending.gate, layout_, time);
    if (pending.controlled_gate == nullptr) return target;

    struct MappedControl {
      unsigned position;
      unsigned value;
    };
    const auto& raw = *pending.controlled_gate;
    std::vector<MappedControl> controls;
    controls.reserve(raw.controlled_by.size());
    for (unsigned i = 0; i < raw.controlled_by.size(); ++i) {
      controls.push_back({layout_.PhysicalPositionOf(raw.controlled_by[i]),
                          unsigned((raw.cmask >> i) & 1)});
    }
    std::sort(
        controls.begin(), controls.end(),
        [](const MappedControl& a, const MappedControl& b) {
          return a.position < b.position;
        });

    ControlledGate<fp_type> physical;
    static_cast<Gate<fp_type>&>(physical) = std::move(target);
    physical.cmask = 0;
    for (const MappedControl& control : controls) {
      physical.cmask |=
          uint64_t{control.value} << physical.controlled_by.size();
      physical.controlled_by.push_back(control.position);
    }
    return physical;
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

  std::optional<std::vector<ExecutableGate>> BuildExecutableGates(
      const std::vector<MappedGate>& batch_gates) {
    // The fuser tracks dependencies on targets and controls, including controls
    // above the tile. Only targets of ordinary gates enter fused matrices, so
    // those matrices still act entirely within the tile.
    auto fused_ops =
        Fuser::FuseGates(param_, zones_.num_qubits, batch_gates);
    if (fused_ops.empty() && !batch_gates.empty()) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return std::nullopt;
    }

    // Controlled gates and unfused multi-qubit gates pass through the fuser as
    // const pointers into batch_gates. Copy their data so executable gates own
    // it after batch_gates expires.
    std::vector<ExecutableGate> executable_gates;
    executable_gates.reserve(fused_ops.size());
    for (auto& operation : fused_ops) {
      if (const auto* controlled =
              OpGetAlternative<ControlledGate<fp_type>>(std::as_const(operation))) {
        executable_gates.push_back(MakeExecutableControlledGate(*controlled));
      } else if (auto* fused =
                     OpGetAlternative<FusedGate<fp_type>>(operation)) {
        executable_gates.push_back(MakeExecutableFusedGate(*fused));
      } else {
        const auto* gate =
            OpGetAlternative<Gate<fp_type>>(std::as_const(operation));
        assert(gate != nullptr);
        executable_gates.push_back(MakeExecutableGate(*gate));
      }
    }
    simulation_stats_.num_executable_gates += executable_gates.size();
    return executable_gates;
  }

  ExecutableGate MakeExecutableFusedGate(FusedGate<fp_type>& fused) const {
    // Fused qubits are sorted, so the last index determines tile containment.
    assert(fused.qubits.size() <= Backend::kMaxGateQubits);
    assert(fused.qubits.empty() || fused.qubits.back() < zones_.tile_qubits);
    return {std::move(fused.qubits), std::move(fused.matrix)};
  }

  ExecutableGate MakeExecutableGate(const Gate<fp_type>& gate) const {
    assert(gate.qubits.size() <= Backend::kMaxGateQubits);
    assert(gate.qubits.empty() || gate.qubits.back() < zones_.tile_qubits);
    return {gate.qubits, gate.matrix};
  }

  // Split controls by physical position after remapping. The backend sees
  // local control indices plus a predicate on the tile index.
  ExecutableGate MakeExecutableControlledGate(
      const ControlledGate<fp_type>& controlled) const {
    ExecutableGate gate{controlled.qubits, controlled.matrix};
    for (unsigned i = 0; i < controlled.controlled_by.size(); ++i) {
      const auto position = controlled.controlled_by[i];
      const auto value = (controlled.cmask >> i) & 1;
      if (position < zones_.tile_qubits) {
        gate.control_values |= value << gate.controls.size();
        gate.controls.push_back(position);
      } else {
        const auto bit = uint64_t{1} << (position - zones_.tile_qubits);
        gate.tile_control_mask |= bit;
        if (value) gate.tile_control_values |= bit;
      }
    }
    return gate;
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

  // Adds the time of backend work (gate, swap, and measurement passes) to
  // `seconds`. Backend calls may return before the work finishes, so at
  // verbosity > 1 this synchronizes before and after to time the work itself;
  // otherwise the time covers only the call.
  template <typename Work>
  auto TimeBackendWork(double& seconds, Work&& work) {
    if (param_.verbosity > 1) backend_.Synchronize();
    const auto start = GetTime();
    if constexpr (std::is_void_v<decltype(work())>) {
      work();
      if (param_.verbosity > 1) backend_.Synchronize();
      seconds += GetTime() - start;
    } else {
      auto result = work();
      if (param_.verbosity > 1) backend_.Synchronize();
      seconds += GetTime() - start;
      return result;
    }
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
        "%u qubit swaps, %u measurements.\n",
        simulation_stats_.num_gate_batches,
        simulation_stats_.num_restore_passes, simulation_stats_.num_raw_gates,
        simulation_stats_.num_executable_gates, simulation_stats_.num_swaps,
        simulation_stats_.num_measurements);
    if (param_.verbosity > 1) {
      IO::messagef(
          "breakdown: prepare %g s, plan %g s, swap passes %g s, fuse %g s, "
          "gate passes %g s, measure %g s.\n",
          simulation_stats_.prepare_seconds, simulation_stats_.plan_seconds,
          simulation_stats_.swap_seconds, simulation_stats_.fuse_seconds,
          simulation_stats_.gate_seconds, simulation_stats_.measure_seconds);
    }
  }

  const Parameter& param_;
  Backend backend_;
  QubitZones zones_;
  QubitLayout& layout_;
  RGen rgen_;

  // Accumulated over the whole run; printed by LogSimulationSummary.
  SimulationStats simulation_stats_;
};

}  // namespace qsim

#endif  // GATE_BATCH_RUNNER_H_
