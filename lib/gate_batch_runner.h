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
// admits each gate whose qubits still fit into the remap slots and that
// commutes past every gate deferred so far. It tries several starting qubit
// sets ("seeds") and keeps the batch with the best score: gates admitted minus
// an estimated swap cost.
//
// Reading guide:
//
//   GateBatchRunner::Run
//     SimulateCircuit
//       (whole state fits one tile) FuseBatchGates, Backend::ExecuteBatch
//       PreparePendingGates          wrap each gate, flag diagonal ones
//       PlaceHotQubitsInHotZone      fill the hot zone once
//       loop ExecuteNextGateBatch
//         GateBatchPlanner::PlanNextGateBatch
//         BuildSwapsBelow, ApplySwapsToState   swap batch qubits into the tile
//         BuildBatchOperations                 rewrite gates to physical qubits
//         FuseBatchGates, Backend::ExecuteBatch
//       RestoreIdentityQubitOrder    only for Run(State&); the QubitMappedState
//                                    overload keeps the permuted layout

#ifndef GATE_BATCH_RUNNER_H_
#define GATE_BATCH_RUNNER_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "gate.h"
#include "matrix.h"
#include "operation_base.h"
#include "qubit_layout.h"
#include "util.h"

namespace qsim {

namespace gate_batch_internal {

// A circuit gate waiting to be scheduled. Points into the caller's circuit;
// `applied` is set once the gate has run in some batch.
template <typename FP>
struct PendingGate {
  const std::vector<unsigned>& Qubits() const { return gate->qubits; }
  std::size_t Arity() const { return gate->qubits.size(); }

  const Gate<FP>* gate = nullptr;
  bool is_diagonal = false;
  bool applied = false;
};

// Buffers reused across batches so the batch loop does not reallocate.
template <typename FP>
struct GateBatchWorkspace {
  std::vector<QubitSwap> swap_pairs;
  std::vector<FusedGate<FP>> executable_gates;
};

// Counters and timings reported by LogSimulationSummary.
struct SimulationStats {
  unsigned num_gate_batches = 0;
  unsigned num_restore_passes = 0;
  unsigned num_raw_gates = 0;
  unsigned num_swaps = 0;
  unsigned num_executable_gates = 0;
  double plan_seconds = 0.0;
  double swap_seconds = 0.0;
  double fuse_seconds = 0.0;
  double gate_seconds = 0.0;
};

// The logical qubits a candidate batch may hold in the tile, grown as gates
// are admitted. Qubits outside the lane and hot zones each use one remap slot.
struct GateBatchQubitSet {
  bool Contains(unsigned logical_qubit) const {
    return contains_qubit[logical_qubit];
  }

  std::vector<char> contains_qubit;
  unsigned remap_slots_used = 0;
};

// The batch chosen by the planner. `uses_qubit` marks only the qubits the
// admitted gates touch, a subset of the qubit set that produced the plan; only
// these are swapped into the tile.
struct GateBatchPlan {
  bool HasGates() const { return !gate_indices.empty(); }
  std::size_t NumGates() const { return gate_indices.size(); }
  bool UsesQubit(unsigned logical_qubit) const {
    return uses_qubit[logical_qubit];
  }

  std::vector<std::size_t> gate_indices;
  std::vector<char> uses_qubit;
  double score = 0.0;
};

template <typename FP>
class GateBatchPlanner {
 public:
  GateBatchPlanner(
      unsigned num_qubits, unsigned requested_tile_qubits, unsigned tile_qubits,
      unsigned lane_qubits, unsigned min_eviction_floor,
      unsigned max_gate_seeds)
      : num_qubits_(num_qubits),
        requested_tile_qubits_(std::min(requested_tile_qubits, num_qubits)),
        tile_qubits_(tile_qubits),
        lane_qubits_(lane_qubits),
        eviction_floor_(ComputeEvictionFloor(
            num_qubits, tile_qubits, lane_qubits, min_eviction_floor)),
        max_gate_seeds_(max_gate_seeds),
        all_gates_blocked_(num_qubits),
        non_diagonal_gates_blocked_(num_qubits) {}

  // Builds a candidate batch from several starting qubit sets ("seeds") and
  // returns the one with the highest score:
  //   - empty: admits gates greedily in circuit order;
  //   - resident: starts from every qubit already in the tile, which favors
  //     batches that need no swaps;
  //   - lookahead: one seed per later pending gate, starting from that gate's
  //     qubits, so a batch can form around work further ahead. The first
  //     pending gate is skipped because the empty seed already starts there.
  GateBatchPlan PlanNextGateBatch(
      const std::vector<PendingGate<FP>>& gates, const QubitLayout& layout) {
    GateBatchPlan best_plan{};
    auto consider = [&](GateBatchQubitSet seed) {
      auto plan = BuildCandidatePlan(gates, layout, seed);
      if (!best_plan.HasGates() || plan.score > best_plan.score) {
        best_plan = std::move(plan);
      }
    };

    consider(MakeEmptyQubitSet());
    consider(MakeResidentQubitSet(layout));

    const unsigned max_seeds = EffectiveGateSeeds(gates.size());
    unsigned seeds_used = 0;
    bool skipped_first_pending = false;
    for (const PendingGate<FP>& gate : gates) {
      if (gate.applied) continue;
      if (!skipped_first_pending) {
        skipped_first_pending = true;
        continue;
      }
      if (seeds_used == max_seeds) break;
      ++seeds_used;
      auto seed = MakeEmptyQubitSet();
      if (TryAdmitQubits(gate.Qubits(), layout, seed)) {
        consider(std::move(seed));
      }
    }
    return best_plan;
  }

  unsigned EvictionFloor() const { return eviction_floor_; }

 private:
  static constexpr unsigned kMinRemapSlots = 2;
  // Plan scores are measured in gates admitted. A batch that needs swaps pays
  // kSwapPassCost once for the swap pass over the state, plus kSwapPairCost
  // per qubit it swaps into the tile.
  static constexpr double kSwapPassCost = 0.5;
  static constexpr double kSwapPairCost = 1.0 / 16.0;

  // Number of lookahead seeds to try. Full-size tiles are memory-bandwidth
  // bound, so they get the whole max_gate_seeds_ budget: better plans mean
  // fewer batches and swap passes. Tiles the backend shrank to feed more
  // threads make each pass cheap, so planning is capped to stay small next to
  // it.
  unsigned EffectiveGateSeeds(std::size_t num_pending_gates) const {
    if (tile_qubits_ >= num_qubits_ || num_pending_gates == 0) {
      return 0;
    }
    if (tile_qubits_ >= requested_tile_qubits_) {
      return max_gate_seeds_;
    }
    return std::min(
        SeedCapFromTileShrink(), SeedCapFromScanWork(num_pending_gates));
  }

  // Halves the seed budget for each qubit the tile was shrunk by.
  unsigned SeedCapFromTileShrink() const {
    const auto tile_shift = requested_tile_qubits_ - tile_qubits_;
    if (tile_shift >= sizeof(unsigned) * 8) return 0;
    return max_gate_seeds_ >> tile_shift;
  }

  // Each seed scans the pending gates about twice; keep the total below the
  // number of SIMD lane groups one tile pass touches.
  unsigned SeedCapFromScanWork(std::size_t num_pending_gates) const {
    const unsigned tile_qubits_above_lanes =
        tile_qubits_ > lane_qubits_ ? tile_qubits_ - lane_qubits_ : 0u;
    const uint64_t lane_groups_per_tile = uint64_t{1}
                                          << tile_qubits_above_lanes;
    const uint64_t scan_work_per_seed = 2 * uint64_t{num_pending_gates};
    return static_cast<unsigned>(std::min<uint64_t>(
        lane_groups_per_tile / scan_work_per_seed, max_gate_seeds_));
  }

  GateBatchQubitSet MakeEmptyQubitSet() const {
    GateBatchQubitSet qubit_set;
    qubit_set.contains_qubit.assign(num_qubits_, 0);
    return qubit_set;
  }

  unsigned RemapSlotCapacity() const { return tile_qubits_ - eviction_floor_; }

  static unsigned ComputeEvictionFloor(
      unsigned num_qubits, unsigned tile_qubits, unsigned lane_qubits,
      unsigned min_eviction_floor) {
    if (tile_qubits == num_qubits) return tile_qubits;

    const auto tile_qubits_above_lanes =
        tile_qubits > lane_qubits ? tile_qubits - lane_qubits : 0u;
    const auto slots_above_min_floor = tile_qubits > min_eviction_floor
                                           ? tile_qubits - min_eviction_floor
                                           : 0u;
    const auto remap_slots = std::min(
        tile_qubits_above_lanes,
        std::max(slots_above_min_floor, kMinRemapSlots));
    return tile_qubits - remap_slots;
  }

  GateBatchQubitSet MakeResidentQubitSet(const QubitLayout& layout) const {
    auto qubit_set = MakeEmptyQubitSet();
    for (unsigned q = 0; q < num_qubits_; ++q) {
      if (layout.PhysicalPositionOf(q) < tile_qubits_) {
        qubit_set.remap_slots_used +=
            AdditionalRemapSlotsFor(q, qubit_set, layout);
        qubit_set.contains_qubit[q] = 1;
      }
    }
    return qubit_set;
  }

  GateBatchPlan BuildCandidatePlan(
      const std::vector<PendingGate<FP>>& gates, const QubitLayout& layout,
      GateBatchQubitSet& qubit_set) {
    GateBatchPlan plan;
    plan.uses_qubit.assign(num_qubits_, 0);
    ClearBlockedQubits();

    for (std::size_t idx = 0; idx < gates.size(); ++idx) {
      const auto& gate = gates[idx];
      if (gate.applied) continue;
      if (!AnyQubitBlocked(gate) &&
          TryAdmitQubits(gate.Qubits(), layout, qubit_set)) {
        plan.gate_indices.push_back(idx);
        for (unsigned q : gate.Qubits()) {
          plan.uses_qubit[q] = 1;
        }
      } else {
        BlockQubits(gate);
      }
    }

    const auto required_swaps = CountSwapsNeeded(plan, layout);
    const auto swap_cost = required_swaps == 0
                               ? 0.0
                               : kSwapPassCost + kSwapPairCost * required_swaps;
    plan.score = plan.NumGates() - swap_cost;
    return plan;
  }

  unsigned AdditionalRemapSlotsFor(
      unsigned q, const GateBatchQubitSet& qubit_set,
      const QubitLayout& layout) const {
    if (qubit_set.Contains(q)) return 0;
    return layout.PhysicalPositionOf(q) < eviction_floor_ ? 0 : 1;
  }

  bool TryAdmitQubits(
      const std::vector<unsigned>& qubits, const QubitLayout& layout,
      GateBatchQubitSet& qubit_set) const {
    unsigned additional_remap_slots = 0;
    for (unsigned q : qubits) {
      additional_remap_slots += AdditionalRemapSlotsFor(q, qubit_set, layout);
    }
    if (qubit_set.remap_slots_used + additional_remap_slots >
        RemapSlotCapacity()) {
      return false;
    }
    for (unsigned q : qubits) qubit_set.contains_qubit[q] = 1;
    qubit_set.remap_slots_used += additional_remap_slots;
    return true;
  }

  unsigned CountSwapsNeeded(
      const GateBatchPlan& plan, const QubitLayout& layout) const {
    unsigned count = 0;
    for (unsigned q = 0; q < num_qubits_; ++q) {
      if (plan.UsesQubit(q) && layout.PhysicalPositionOf(q) >= tile_qubits_) {
        ++count;
      }
    }
    return count;
  }

  void ClearBlockedQubits() {
    std::fill(all_gates_blocked_.begin(), all_gates_blocked_.end(), 0);
    std::fill(
        non_diagonal_gates_blocked_.begin(), non_diagonal_gates_blocked_.end(),
        0);
  }

  bool AnyQubitBlocked(const PendingGate<FP>& gate) const {
    for (unsigned q : gate.Qubits()) {
      if (all_gates_blocked_[q]) return true;
      if (!gate.is_diagonal && non_diagonal_gates_blocked_[q]) return true;
    }
    return false;
  }

  void BlockQubits(const PendingGate<FP>& gate) {
    for (unsigned q : gate.Qubits()) {
      if (!gate.is_diagonal) all_gates_blocked_[q] = 1;
      non_diagonal_gates_blocked_[q] = 1;
    }
  }

  unsigned num_qubits_;
  unsigned requested_tile_qubits_;
  unsigned tile_qubits_;
  unsigned lane_qubits_;
  unsigned eviction_floor_;
  unsigned max_gate_seeds_;
  // Indexed by logical qubit. A deferred gate blocks later gates on its qubits,
  // except that diagonal gates commute with each other.
  std::vector<char> all_gates_blocked_;  // set by deferred non-diagonal gates
  std::vector<char> non_diagonal_gates_blocked_;  // set by any deferred gate
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

    unsigned tile_qubits = Backend::kDefaultTileQubits;
    unsigned min_eviction_floor = Backend::kDefaultEvictionFloor;
    unsigned max_gate_seeds = 64;
  };

  template <typename Circuit>
  static bool Run(
      const Parameter& param, const Circuit& circuit, State& state) {
    if (!SizesMatch(circuit.num_qubits, state.num_qubits())) return false;
    QubitLayout layout(circuit.num_qubits);
    GateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuit(circuit, true);
  }

  template <typename Circuit>
  static bool Run(
      const Parameter& param, const Circuit& circuit, QubitMappedState& state) {
    if (!SizesMatch(circuit.num_qubits, state.state.num_qubits()) ||
        !SizesMatch(circuit.num_qubits, state.layout.NumQubits())) {
      return false;
    }
    GateBatchRunner runner(
        param, circuit.num_qubits, state.state, state.layout);
    return runner.SimulateCircuit(circuit, false);
  }

 private:
  using PendingGate = gate_batch_internal::PendingGate<fp_type>;
  using SimulationStats = gate_batch_internal::SimulationStats;
  using GateBatchWorkspace = gate_batch_internal::GateBatchWorkspace<fp_type>;
  using GateBatchPlanner = gate_batch_internal::GateBatchPlanner<fp_type>;
  using GateBatchPlan = gate_batch_internal::GateBatchPlan;

  GateBatchRunner(
      const Parameter& param, unsigned num_qubits, State& state,
      QubitLayout& layout)
      : param_(param),
        num_qubits_(num_qubits),
        backend_(param, num_qubits, state),
        lane_qubits_(backend_.LaneQubits()),
        layout_(layout),
        gate_batch_planner_(
            num_qubits, backend_.RequestedTileQubits(), backend_.TileQubits(),
            lane_qubits_, param.min_eviction_floor, param.max_gate_seeds) {
    assert(layout_.NumQubits() == num_qubits_);
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
  bool SimulateCircuit(const Circuit& circuit, bool restore_qubit_order) {
    using Op = typename std::decay_t<decltype(circuit.ops)>::value_type;

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
    if (!backend_.Prepare()) return false;
    simulation_stats_.num_raw_gates = unsigned(circuit.ops.size());

    if (backend_.TileQubits() == num_qubits_) {
      if (!ValidateCircuitOperations(circuit)) return false;
      const auto fuse_start = GetTime();
      if (!FuseBatchGates(circuit.ops)) return false;
      simulation_stats_.fuse_seconds += GetTime() - fuse_start;
      simulation_stats_.num_executable_gates +=
          gate_batch_workspace_.executable_gates.size();
      backend_.Synchronize();
      const auto simulation_start = GetTime();
      const auto gates_start = StartTimer();
      backend_.ExecuteBatch(gate_batch_workspace_.executable_gates);
      simulation_stats_.gate_seconds += StopTimer(gates_start);
      simulation_stats_.num_gate_batches = circuit.ops.empty() ? 0 : 1;
      backend_.Synchronize();
      LogSimulationSummary(simulation_start);
      return true;
    }

    const auto prepare_start = GetTime();
    if (!PreparePendingGates(circuit)) return false;
    LogPreparationTime(prepare_start);

    backend_.Synchronize();
    const auto simulation_start = GetTime();
    PlaceHotQubitsInHotZone();

    std::vector<Op> batch_operations;

    std::size_t applied_count = 0;
    while (applied_count < pending_gates_.size()) {
      const auto num_applied = ExecuteNextGateBatch(batch_operations);
      if (num_applied == 0) return false;  // fuser error or stuck schedule
      applied_count += num_applied;
    }

    if (restore_qubit_order) RestoreIdentityQubitOrder();
    backend_.Synchronize();
    LogSimulationSummary(simulation_start);
    return true;
  }

  template <typename Op>
  std::size_t ExecuteNextGateBatch(std::vector<Op>& batch_operations) {
    const auto tile_qubits = backend_.TileQubits();

    const auto plan_start = GetTime();
    const auto plan =
        gate_batch_planner_.PlanNextGateBatch(pending_gates_, layout_);
    simulation_stats_.plan_seconds += GetTime() - plan_start;
    if (!plan.HasGates()) {
      IO::errorf(
          "qsim_gate_batch: a gate does not fit the qubit set "
          "of %u tile qubits; use a larger tile_qubits.\n",
          tile_qubits);
      return 0;
    }

    BuildSwapsBelow(
        plan.uses_qubit, tile_qubits, gate_batch_workspace_.swap_pairs);
    ApplySwapsToState(gate_batch_workspace_.swap_pairs);

    const auto fuse_start = GetTime();
    BuildBatchOperations(plan, batch_operations);
    if (!FuseBatchGates(batch_operations)) {
      return 0;
    }
    simulation_stats_.fuse_seconds += GetTime() - fuse_start;
    simulation_stats_.num_executable_gates +=
        gate_batch_workspace_.executable_gates.size();
    MarkGatesApplied(plan);

    const auto gates_start = StartTimer();
    backend_.ExecuteBatch(gate_batch_workspace_.executable_gates);
    simulation_stats_.gate_seconds += StopTimer(gates_start);

    ++simulation_stats_.num_gate_batches;
    return plan.NumGates();
  }

  static void NormalizeGateQubitOrder(
      std::vector<unsigned>& qubits, Matrix<fp_type>& matrix) {
    if (qubits.size() < 2) return;
    auto permutation = NormalToGateOrderPermutation(qubits);
    if (!permutation.empty()) {
      MatrixShuffle(permutation, qubits.size(), matrix);
      std::sort(qubits.begin(), qubits.end());
    }
  }

  static bool IsDiagonalMatrix(
      const Matrix<fp_type>& matrix, std::size_t arity) {
    const auto dim = std::size_t{1} << arity;
    if (matrix.size() < 2 * dim * dim) return false;

    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = 0; col < dim; ++col) {
        if (row == col) continue;
        const auto k = 2 * (row * dim + col);
        if (matrix[k] != 0 || matrix[k + 1] != 0) return false;
      }
    }
    return true;
  }

  template <typename Circuit>
  bool ValidateCircuitOperations(const Circuit& circuit) const {
    for (const auto& operation : circuit.ops) {
      const auto* raw_gate = OpGetAlternative<Gate<fp_type>>(operation);
      if (raw_gate == nullptr) {
        IO::errorf(
            "qsim_gate_batch: unsupported operation "
            "(controlled gate or measurement).\n");
        return false;
      }
      if (raw_gate->qubits.size() > Backend::kMaxGateQubits) {
        IO::errorf(
            "qsim_gate_batch: %zu-qubit gate exceeds the backend "
            "limit of %u qubits.\n",
            raw_gate->qubits.size(), Backend::kMaxGateQubits);
        return false;
      }
    }
    return true;
  }

  template <typename Circuit>
  bool PreparePendingGates(const Circuit& circuit) {
    pending_gates_.reserve(circuit.ops.size());

    for (const auto& operation : circuit.ops) {
      const auto* raw_gate = OpGetAlternative<Gate<fp_type>>(operation);
      if (raw_gate == nullptr) {
        IO::errorf(
            "qsim_gate_batch: unsupported operation "
            "(controlled gate or measurement).\n");
        return false;
      }
      if (raw_gate->qubits.size() > Backend::kMaxGateQubits) {
        IO::errorf(
            "qsim_gate_batch: %zu-qubit gate exceeds the backend "
            "limit of %u qubits.\n",
            raw_gate->qubits.size(), Backend::kMaxGateQubits);
        return false;
      }

      const bool diagonal =
          IsDiagonalMatrix(raw_gate->matrix, raw_gate->qubits.size());
      pending_gates_.push_back({raw_gate, diagonal});
    }
    return true;
  }

  std::vector<uint64_t> ComputeQubitUsageScores() const {
    std::vector<uint64_t> scores(num_qubits_, 0);
    for (const PendingGate& gate : pending_gates_) {
      const auto weight = gate.Arity();
      for (unsigned q : gate.Qubits()) scores[q] += weight;
    }
    return scores;
  }

  static std::vector<char> SelectHighestScoringQubits(
      const std::vector<uint64_t>& scores, unsigned first_qubit,
      unsigned count) {
    std::vector<unsigned> candidates(scores.size() - first_qubit);
    std::iota(candidates.begin(), candidates.end(), first_qubit);
    assert(count <= candidates.size());
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&scores](unsigned a, unsigned b) { return scores[a] > scores[b]; });

    std::vector<char> selected(scores.size(), 0);
    for (unsigned i = 0; i < count; ++i) selected[candidates[i]] = 1;
    return selected;
  }

  void PlaceHotQubitsInHotZone() {
    const auto eviction_floor = gate_batch_planner_.EvictionFloor();
    if (backend_.TileQubits() == num_qubits_ ||
        eviction_floor <= lane_qubits_) {
      return;
    }

    const auto usage_scores = ComputeQubitUsageScores();
    const auto is_hot = SelectHighestScoringQubits(
        usage_scores, lane_qubits_, eviction_floor - lane_qubits_);

    auto& swap_pairs = gate_batch_workspace_.swap_pairs;
    BuildSwapsBelow(is_hot, eviction_floor, swap_pairs);
    LogHotZonePlacement(usage_scores, eviction_floor, swap_pairs.size());
    ApplySwapsToState(swap_pairs);
  }

  void BuildSwapsBelow(
      const std::vector<char>& wanted, unsigned boundary,
      std::vector<QubitSwap>& swaps) {
    swaps.clear();
    auto evict_position = boundary;

    for (unsigned q = 0; q < wanted.size(); ++q) {
      if (!wanted[q]) continue;
      const auto position = layout_.PhysicalPositionOf(q);
      if (position < boundary) continue;

      do {
        assert(evict_position > lane_qubits_);
        --evict_position;
      } while (wanted[layout_.LogicalQubitAt(evict_position)]);
      swaps.emplace_back(evict_position, position);
      layout_.SwapPhysicalPositions(evict_position, position);
    }
  }

  void ApplySwapsToState(const std::vector<QubitSwap>& swap_pairs) {
    if (swap_pairs.empty()) return;
    const auto swap_start = StartTimer();
    backend_.ApplySwaps(swap_pairs);
    simulation_stats_.swap_seconds += StopTimer(swap_start);
    simulation_stats_.num_swaps += swap_pairs.size();
  }

  template <typename Op>
  void BuildBatchOperations(
      const GateBatchPlan& plan, std::vector<Op>& batch_operations) const {
    batch_operations.clear();
    batch_operations.reserve(plan.NumGates());

    unsigned time = 0;
    for (std::size_t idx : plan.gate_indices) {
      const auto& raw_gate = *pending_gates_[idx].gate;

      std::vector<unsigned> physical_qubits(raw_gate.qubits.size());
      for (std::size_t i = 0; i < raw_gate.qubits.size(); ++i) {
        physical_qubits[i] = layout_.PhysicalPositionOf(raw_gate.qubits[i]);
      }
      auto physical_matrix = raw_gate.matrix;
      NormalizeGateQubitOrder(physical_qubits, physical_matrix);

      auto physical_gate = Gate<fp_type>{
          {0, time++, std::move(physical_qubits)},
          {},
          std::move(physical_matrix),
          false};
      batch_operations.push_back(Op{std::move(physical_gate)});
    }
  }

  template <typename Op>
  bool FuseBatchGates(const std::vector<Op>& batch_operations) {
    auto fused_ops =
        Fuser::FuseGates(param_, backend_.TileQubits(), batch_operations);
    if (fused_ops.empty() && !batch_operations.empty()) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return false;
    }

    auto& executable_gates = gate_batch_workspace_.executable_gates;
    executable_gates.clear();
    executable_gates.reserve(fused_ops.size());
    // Fused ops reference input batch_operations, so copying must happen while
    // it is alive.
    for (auto& operation : fused_ops) {
      auto* gate = OpGetAlternative<FusedGate<fp_type>>(operation);
      if (gate == nullptr) {
        IO::errorf("qsim_gate_batch: unsupported operation in gate batch.\n");
        return false;
      }
      assert(std::is_sorted(gate->qubits.begin(), gate->qubits.end()));
      executable_gates.push_back(std::move(*gate));
    }
    return true;
  }

  void MarkGatesApplied(const GateBatchPlan& plan) {
    for (std::size_t idx : plan.gate_indices) {
      pending_gates_[idx].applied = true;
    }
  }

  void RestoreIdentityQubitOrder() {
    std::vector<QubitSwap> swap_pairs;

    while (layout_.BuildIdentityRestorationSwaps(swap_pairs)) {
      ApplySwapsToState(swap_pairs);
      ++simulation_stats_.num_restore_passes;
    }
  }

  double StartTimer() {
    if (param_.verbosity > 1) backend_.Synchronize();
    return GetTime();
  }

  double StopTimer(double start) {
    if (param_.verbosity > 1) backend_.Synchronize();
    return GetTime() - start;
  }

  void LogPreparationTime(double prepare_start) const {
    if (param_.verbosity <= 1) return;
    IO::messagef("prepare time is %g seconds.\n", GetTime() - prepare_start);
  }

  void LogHotZonePlacement(
      const std::vector<uint64_t>& usage_scores, unsigned eviction_floor,
      std::size_t num_initial_swaps) const {
    if (param_.verbosity <= 1) return;

    IO::messagef("hot zone [%u,%u):", lane_qubits_, eviction_floor);
    for (unsigned p = lane_qubits_; p < eviction_floor; ++p) {
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
          "breakdown: plan %g s, swap passes %g s, fuse %g s, "
          "gate passes %g s.\n",
          simulation_stats_.plan_seconds, simulation_stats_.swap_seconds,
          simulation_stats_.fuse_seconds, simulation_stats_.gate_seconds);
    }
  }

  const Parameter& param_;
  const unsigned num_qubits_;
  Backend backend_;
  const unsigned lane_qubits_;
  std::vector<PendingGate> pending_gates_;
  QubitLayout& layout_;
  GateBatchPlanner gate_batch_planner_;
  GateBatchWorkspace gate_batch_workspace_;
  SimulationStats simulation_stats_;
};

}  // namespace qsim

#endif  // GATE_BATCH_RUNNER_H_
