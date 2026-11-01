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

// Combinatorial circuit preparation and lookahead batch planning for
// GateBatchRunner.

#ifndef GATE_BATCH_PLANNER_H_
#define GATE_BATCH_PLANNER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "gate.h"
#include "operation_base.h"
#include "qubit_layout.h"

namespace qsim {

namespace gate_batch_internal {

// The zone boundaries of the physical bit-position layout, fixed for the whole
// run:
//
// 0           lane_qubits      eviction_floor     tile_qubits       num_qubits
// | lane zone |    hot zone    |    remap zone    | tile index bits |
struct QubitZones {
  template <typename Backend>
  static QubitZones ForBackend(
      const Backend& backend, unsigned num_qubits,
      unsigned min_eviction_floor) {
    QubitZones zones;
    zones.num_qubits = num_qubits;
    zones.tile_qubits = backend.TileQubits();
    zones.lane_qubits = backend.LaneQubits();
    zones.eviction_floor = zones.ComputeEvictionFloor(min_eviction_floor);
    return zones;
  }

  bool IsSingleTile() const { return tile_qubits == num_qubits; }
  unsigned NumHotSlots() const { return eviction_floor - lane_qubits; }
  unsigned NumRemapSlots() const { return tile_qubits - eviction_floor; }
  unsigned TileQubitsAboveLanes() const {
    return tile_qubits > lane_qubits ? tile_qubits - lane_qubits : 0u;
  }

  unsigned lane_qubits = 0;     // end of the lane zone
  unsigned eviction_floor = 0;  // end of the hot zone
  unsigned tile_qubits = 0;     // end of the remap zone
  unsigned num_qubits = 0;

 private:
  static constexpr unsigned kMinRemapSlots = 2;

  unsigned ComputeEvictionFloor(unsigned min_eviction_floor) const {
    if (IsSingleTile()) return tile_qubits;

    const auto slots_above_min_floor = tile_qubits > min_eviction_floor
                                           ? tile_qubits - min_eviction_floor
                                           : 0u;
    const auto remap_slots = std::min(
        TileQubitsAboveLanes(),
        std::max(slots_above_min_floor, kMinRemapSlots));
    return tile_qubits - remap_slots;
  }
};

// A circuit gate to be scheduled. Points into the caller's circuit; `index` is
// its position in the circuit.
template <typename FP>
struct PendingGate {
  PendingGate(const Gate<FP>& raw_gate, std::size_t index)
      : gate(&raw_gate), index(index), is_diagonal(IsDiagonalGate(raw_gate)) {}

  PendingGate(const ControlledGate<FP>& controlled, std::size_t index)
      : PendingGate(static_cast<const Gate<FP>&>(controlled), index) {
    controlled_gate = &controlled;
    all_qubits = controlled.qubits;
    all_qubits.insert(all_qubits.end(), controlled.controlled_by.begin(),
                      controlled.controlled_by.end());
  }

  // Targets need tile slots; controls only constrain execution and ordering.
  const qsim::Qubits& TargetQubits() const { return gate->qubits; }
  const qsim::Qubits& Qubits() const {
    return controlled_gate == nullptr ? TargetQubits() : all_qubits;
  }

  const Gate<FP>* gate;
  std::size_t index;
  bool is_diagonal;
  const ControlledGate<FP>* controlled_gate = nullptr;
  qsim::Qubits all_qubits;

 private:
  // Whether the gate's matrix is diagonal. Two diagonal gates give the same
  // result in either order, so the planner may swap their order.
  static bool IsDiagonalGate(const Gate<FP>& gate) {
    const auto& matrix = gate.matrix;
    const auto dim = std::size_t{1} << gate.qubits.size();
    assert(matrix.size() == 2 * dim * dim);

    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = 0; col < dim; ++col) {
        if (row == col) continue;
        const auto k = 2 * (row * dim + col);
        const auto re = matrix[k];
        const auto im = matrix[k + 1];
        if (re != 0 || im != 0) return false;
      }
    }
    return true;
  }
};

// The circuit's gates in order, each marked applied once it has run. The list
// never changes after FromCircuit. All gate pointers refer to the caller's
// circuit, which must outlive this list.
template <typename FP>
class PendingGateList {
 public:
  using PendingGate = gate_batch_internal::PendingGate<FP>;

  PendingGateList() = default;
  PendingGateList(PendingGateList&&) = default;
  PendingGateList& operator=(PendingGateList&&) = default;
  PendingGateList(const PendingGateList&) = delete;
  PendingGateList& operator=(const PendingGateList&) = delete;

  // Controls do not enlarge the target matrix or count toward the backend's
  // target-width limit. Rejects measurements and other operation types.
  template <typename Circuit>
  static std::optional<PendingGateList> FromCircuit(
      const Circuit& circuit, unsigned max_gate_qubits,
      unsigned max_controlled_gate_qubits) {
    PendingGateList list;
    list.gates_.reserve(circuit.ops.size());

    for (const auto& operation : circuit.ops) {
      const Gate<FP>* gate = OpGetAlternative<Gate<FP>>(operation);
      if (const auto* controlled =
              OpGetAlternative<ControlledGate<FP>>(operation)) {
        if (controlled->qubits.size() > max_controlled_gate_qubits) {
          return std::nullopt;
        }
        list.gates_.emplace_back(*controlled, list.gates_.size());
        continue;
      }
      if (gate == nullptr || gate->qubits.size() > max_gate_qubits) {
        return std::nullopt;
      }
      list.gates_.emplace_back(*gate, list.gates_.size());
    }
    list.applied_.assign(list.gates_.size(), 0);
    return list;
  }

  std::size_t Size() const { return gates_.size(); }
  const PendingGate& operator[](std::size_t i) const { return gates_[i]; }

  // The first gate after `gate` not yet applied, or the first one at all when
  // `gate` is nullptr. Returns nullptr once no gate is pending.
  const PendingGate* NextPending(const PendingGate* gate = nullptr) const {
    return PendingAtOrAfter(gate == nullptr ? first_pending_ : gate->index + 1);
  }

  void MarkApplied(const std::vector<std::size_t>& indices) {
    for (std::size_t i : indices) {
      assert(!applied_[i]);
      applied_[i] = 1;
    }
    while (first_pending_ < gates_.size() && applied_[first_pending_]) {
      ++first_pending_;
    }
  }

 private:
  const PendingGate* PendingAtOrAfter(std::size_t index) const {
    while (index < gates_.size() && applied_[index]) ++index;
    return index < gates_.size() ? &gates_[index] : nullptr;
  }

  std::vector<PendingGate> gates_;
  std::vector<char> applied_;
  // Every gate before this index is applied.
  std::size_t first_pending_ = 0;
};

// The batch chosen by the planner. `uses_qubit` marks the qubits the admitted
// gates target; those above the tile are swapped in before the batch runs.
struct GateBatchPlan {
  explicit GateBatchPlan(unsigned num_qubits) : uses_qubit(num_qubits, 0) {}

  // Adds the gate to the batch and marks its target qubits as used.
  template <typename FP>
  void AddGate(const PendingGate<FP>& gate) {
    gate_indices.push_back(gate.index);
    for (unsigned q : gate.TargetQubits()) uses_qubit[q] = 1;
  }

  bool HasGates() const { return !gate_indices.empty(); }
  std::size_t NumGates() const { return gate_indices.size(); }

  // Measured in gates admitted. A batch that needs swaps pays kSwapPassCost
  // once for the swap pass over the state, plus kSwapPairCost per qubit it
  // swaps into the tile.
  double Score() const {
    constexpr double kSwapPassCost = 0.5;
    constexpr double kSwapPairCost = 1.0 / 16.0;
    const auto swap_cost =
        num_swaps == 0 ? 0.0 : kSwapPassCost + kSwapPairCost * num_swaps;
    return NumGates() - swap_cost;
  }

  std::vector<std::size_t> gate_indices;
  std::vector<char> uses_qubit;
  // Qubits the batch uses that are stored above the tile; set by the planner.
  unsigned num_swaps = 0;
};

template <typename FP>
class GateBatchPlanner {
 public:
  using PendingGateList = gate_batch_internal::PendingGateList<FP>;

  // Reads `pending_gates` and `layout` at each PlanNextGateBatch call, so it
  // always plans from the current state; both must outlive the planner.
  GateBatchPlanner(
      const QubitZones& zones, const PendingGateList& pending_gates,
      const QubitLayout& layout, unsigned max_lookahead_seeds)
      : zones_(zones),
        pending_gates_(pending_gates),
        layout_(layout),
        max_lookahead_seeds_(max_lookahead_seeds) {}

  // A seed is the set of remap slots a candidate batch starts with; admitting a
  // gate adds its targets that need slots. Builds one candidate per seed and
  // returns the one with the highest score (the first one built on a tie).
  // There is one seed per pending gate, starting from the slots that gate
  // needs, so a batch can form around work further ahead. The first pending
  // gate's seed is plain greedy, since the walk admits that gate first anyway;
  // it does not count against the lookahead budget.
  GateBatchPlan PlanNextGateBatch() const {
    GateBatchPlan best_plan(zones_.num_qubits);
    unsigned seeds_used = 0;  // `<=` below: the first, greedy seed is free
    for (const auto* gate = pending_gates_.NextPending();
         gate != nullptr && seeds_used <= max_lookahead_seeds_;
         gate = pending_gates_.NextPending(gate), ++seeds_used) {
      RemapSlotSet seed(zones_.num_qubits);
      if (!TryAdmitQubits(gate->TargetQubits(), seed)) continue;
      auto plan = BuildCandidatePlan(seed);
      if (!best_plan.HasGates() || plan.Score() > best_plan.Score()) {
        best_plan = std::move(plan);
      }
    }
    return best_plan;
  }

 private:
  // The logical qubits holding a remap slot in a candidate batch, grown as
  // gates are admitted. Qubits below the eviction floor stay in the tile for
  // free and are never added.
  class RemapSlotSet {
   public:
    explicit RemapSlotSet(unsigned num_qubits)
        : contains_qubit_(num_qubits, 0) {}

    bool Contains(unsigned logical_qubit) const {
      return contains_qubit_[logical_qubit];
    }
    unsigned NumSlotsUsed() const { return num_slots_used_; }

    void Add(unsigned logical_qubit) {
      if (Contains(logical_qubit)) return;
      contains_qubit_[logical_qubit] = 1;
      ++num_slots_used_;
    }

   private:
    std::vector<char> contains_qubit_;
    unsigned num_slots_used_ = 0;
  };

  // What the gates deferred so far in a candidate batch block, per logical
  // qubit. A later gate may not move past a deferred gate on a shared qubit,
  // except that two diagonal gates may swap order.
  class DeferredGates {
   public:
    explicit DeferredGates(unsigned num_qubits)
        : blocks_(num_qubits, Level::kNone) {}

    // Whether the gate may run before every gate deferred so far.
    bool CanRunAhead(const PendingGate<FP>& gate) const {
      const auto blocking =
          gate.is_diagonal ? Level::kAll : Level::kNonDiagonal;
      for (unsigned q : gate.Qubits()) {
        if (blocks_[q] >= blocking) return false;
      }
      return true;
    }

    void Add(const PendingGate<FP>& gate) {
      const auto level = gate.is_diagonal ? Level::kNonDiagonal : Level::kAll;
      for (unsigned q : gate.Qubits()) {
        blocks_[q] = std::max(blocks_[q], level);
      }
    }

   private:
    // Which later gates a qubit blocks; each level includes the one below.
    enum class Level : char { kNone, kNonDiagonal, kAll };

    std::vector<Level> blocks_;
  };

  GateBatchPlan BuildCandidatePlan(RemapSlotSet& slots) const {
    GateBatchPlan plan(zones_.num_qubits);
    DeferredGates deferred(zones_.num_qubits);

    for (const auto* gate = pending_gates_.NextPending(); gate != nullptr;
         gate = pending_gates_.NextPending(gate)) {
      if (deferred.CanRunAhead(*gate) &&
          TryAdmitQubits(gate->TargetQubits(), slots)) {
        plan.AddGate(*gate);
      } else {
        deferred.Add(*gate);
      }
    }

    plan.num_swaps = CountSwapsNeeded(plan);
    return plan;
  }

  bool NeedsRemapSlot(unsigned q) const {
    return layout_.PhysicalPositionOf(q) >= zones_.eviction_floor;
  }

  // Adds the gate's qubits to `slots` if the remap zone has room for them.
  bool TryAdmitQubits(const Qubits& qubits, RemapSlotSet& slots) const {
    unsigned new_slots = 0;
    for (unsigned q : qubits) {
      if (!slots.Contains(q) && NeedsRemapSlot(q)) ++new_slots;
    }
    if (slots.NumSlotsUsed() + new_slots > zones_.NumRemapSlots()) return false;

    for (unsigned q : qubits) {
      if (NeedsRemapSlot(q)) slots.Add(q);
    }
    return true;
  }

  unsigned CountSwapsNeeded(const GateBatchPlan& plan) const {
    unsigned count = 0;
    for (unsigned q = 0; q < zones_.num_qubits; ++q) {
      if (plan.uses_qubit[q] &&
          layout_.PhysicalPositionOf(q) >= zones_.tile_qubits) {
        ++count;
      }
    }
    return count;
  }

  const QubitZones zones_;
  const PendingGateList& pending_gates_;
  const QubitLayout& layout_;
  unsigned max_lookahead_seeds_;
};

}  // namespace gate_batch_internal

}  // namespace qsim

#endif  // GATE_BATCH_PLANNER_H_
