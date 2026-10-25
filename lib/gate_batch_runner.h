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
//       SimulateSingleTile           whole state fits one tile:
//         FuseBatchGates, ExecuteFusedGates  fuse the circuit, run it once
//       SimulateMultiTile
//         PreparePendingGates        wrap each gate, flag diagonal ones
//         FillHotZone                move the most-used qubits there, once
//         loop ExecuteNextGateBatch
//           GateBatchPlanner::PlanNextGateBatch
//           MoveQubitsIntoZone       swap batch qubits into the remap zone
//           BuildBatchOperations     rewrite gates to physical qubits
//           FuseBatchGates, ExecuteFusedGates
//         RestoreIdentityQubitOrder  only for Run(State&); the
//                                    QubitMappedState overload keeps the
//                                    permuted layout

#ifndef GATE_BATCH_RUNNER_H_
#define GATE_BATCH_RUNNER_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
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

// --- Qubit zones ---

// The zone boundaries from the diagram at the top, fixed for the whole run.
struct QubitZones {
  template <typename Backend>
  static QubitZones ForBackend(
      const Backend& backend, unsigned num_qubits,
      unsigned min_eviction_floor) {
    QubitZones zones;
    zones.num_qubits = num_qubits;
    zones.requested_tile_qubits =
        std::min(backend.RequestedTileQubits(), num_qubits);
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
  // The tile size before the backend shrank it to give every thread a tile.
  unsigned requested_tile_qubits = 0;

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

// --- Pending gates ---

// Whether the gate's matrix is diagonal. Two diagonal gates give the same
// result in either order, so the planner may swap their order.
template <typename FP>
bool IsDiagonalGate(const Gate<FP>& gate) {
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

// A circuit gate to be scheduled. Points into the caller's circuit; `index` is
// its position in the circuit.
template <typename FP>
struct PendingGate {
  PendingGate(const Gate<FP>& raw_gate, std::size_t index)
      : gate(&raw_gate), index(index), is_diagonal(IsDiagonalGate(raw_gate)) {}

  const std::vector<unsigned>& Qubits() const { return gate->qubits; }

  const Gate<FP>* gate;
  std::size_t index;
  bool is_diagonal;
};

// The circuit's gates in order, each marked applied once it has run. The list
// never changes after FromCircuit, so gate indices and the pointers handed out
// by NextPending stay valid for the whole run.
template <typename FP>
class PendingGateList {
 public:
  using PendingGate = gate_batch_internal::PendingGate<FP>;

  // Returns nullopt if the circuit has an operation other than a plain gate.
  template <typename Circuit>
  static std::optional<PendingGateList> FromCircuit(const Circuit& circuit) {
    PendingGateList list;
    list.gates_.reserve(circuit.ops.size());
    for (const auto& operation : circuit.ops) {
      const auto* raw_gate = OpGetAlternative<Gate<FP>>(operation);
      if (raw_gate == nullptr) return std::nullopt;
      list.gates_.emplace_back(*raw_gate, list.gates_.size());
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

// --- Batch planning ---

// The logical qubits holding a remap slot in a candidate batch, grown as gates
// are admitted. Qubits below the eviction floor stay in the tile for free and
// are never added.
class RemapSlotSet {
 public:
  explicit RemapSlotSet(unsigned num_qubits) : contains_qubit_(num_qubits, 0) {}

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

  template <typename FP>
  bool Blocks(const PendingGate<FP>& gate) const {
    const auto blocking = gate.is_diagonal ? Level::kAll : Level::kNonDiagonal;
    for (unsigned q : gate.Qubits()) {
      if (blocks_[q] >= blocking) return true;
    }
    return false;
  }

  template <typename FP>
  void Defer(const PendingGate<FP>& gate) {
    const auto level = gate.is_diagonal ? Level::kNonDiagonal : Level::kAll;
    for (unsigned q : gate.Qubits()) blocks_[q] = std::max(blocks_[q], level);
  }

 private:
  // Which later gates a qubit blocks; each level includes the one below.
  enum class Level : char { kNone, kNonDiagonal, kAll };

  std::vector<Level> blocks_;
};

// The batch chosen by the planner. `uses_qubit` marks the qubits the admitted
// gates touch; those above the tile are swapped in before the batch runs.
struct GateBatchPlan {
  explicit GateBatchPlan(unsigned num_qubits) : uses_qubit(num_qubits, 0) {}

  // Adds the gate to the batch and marks its qubits as used.
  template <typename FP>
  void AddGate(const PendingGate<FP>& gate) {
    gate_indices.push_back(gate.index);
    for (unsigned q : gate.Qubits()) uses_qubit[q] = 1;
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

  GateBatchPlanner(const QubitZones& zones, unsigned max_lookahead_seeds)
      : zones_(zones), max_lookahead_seeds_(max_lookahead_seeds) {}

  // A seed is the set of remap slots a candidate batch starts with; admitting a
  // gate adds its qubits that need slots. Builds one candidate per seed and
  // returns the one with the highest score. Seeds:
  //   - empty: admits gates greedily in circuit order;
  //   - resident: starts from every qubit already in the remap zone, which
  //     favors batches that need no swaps;
  //   - lookahead: one seed per later pending gate, starting from that gate's
  //     qubits, so a batch can form around work further ahead.
  GateBatchPlan PlanNextGateBatch(
      const PendingGateList& gates, const QubitLayout& layout) const {
    GateBatchPlan best_plan(zones_.num_qubits);
    auto consider = [&](RemapSlotSet seed) {
      auto plan = BuildCandidatePlan(gates, layout, seed);
      if (!best_plan.HasGates() || plan.Score() > best_plan.Score()) {
        best_plan = std::move(plan);
      }
    };

    consider(RemapSlotSet(zones_.num_qubits));
    consider(MakeResidentSeed(layout));

    // A seed for the first pending gate would rebuild the empty seed's plan, so
    // lookahead starts at the gate after it.
    const unsigned num_lookahead_seeds = NumLookaheadSeeds(gates.Size());
    unsigned seeds_used = 0;
    const auto* first_pending = gates.NextPending();
    assert(
        first_pending != nullptr);  // the runner plans only while gates remain
    for (const auto* gate = gates.NextPending(first_pending);
         gate != nullptr && seeds_used < num_lookahead_seeds;
         gate = gates.NextPending(gate)) {
      ++seeds_used;
      RemapSlotSet seed(zones_.num_qubits);
      if (TryAdmitQubits(gate->Qubits(), layout, seed)) {
        consider(std::move(seed));
      }
    }
    return best_plan;
  }

 private:
  // Number of lookahead seeds to try. Full-size tiles are memory-bandwidth
  // bound, so they get the whole max_lookahead_seeds_ budget: better plans mean
  // fewer batches and swap passes. Tiles the backend shrank to feed more
  // threads make each pass cheap, so planning is capped to stay small next to
  // it.
  unsigned NumLookaheadSeeds(std::size_t num_pending_gates) const {
    if (zones_.IsSingleTile() || num_pending_gates == 0) {
      return 0;
    }
    if (zones_.tile_qubits >= zones_.requested_tile_qubits) {
      return max_lookahead_seeds_;
    }
    return std::min(
        SeedCapFromTileShrink(), SeedCapFromScanWork(num_pending_gates));
  }

  // Halves the seed budget for each qubit the tile was shrunk by.
  unsigned SeedCapFromTileShrink() const {
    const auto tile_shift = zones_.requested_tile_qubits - zones_.tile_qubits;
    if (tile_shift >= sizeof(unsigned) * 8) return 0;
    return max_lookahead_seeds_ >> tile_shift;
  }

  // Each seed scans the pending gates about twice; keep the total below the
  // number of SIMD lane groups one tile pass touches.
  unsigned SeedCapFromScanWork(std::size_t num_pending_gates) const {
    const uint64_t lane_groups_per_tile = uint64_t{1}
                                          << zones_.TileQubitsAboveLanes();
    const uint64_t scan_work_per_seed = 2 * uint64_t{num_pending_gates};
    return static_cast<unsigned>(std::min<uint64_t>(
        lane_groups_per_tile / scan_work_per_seed, max_lookahead_seeds_));
  }

  RemapSlotSet MakeResidentSeed(const QubitLayout& layout) const {
    RemapSlotSet seed(zones_.num_qubits);
    for (unsigned q = 0; q < zones_.num_qubits; ++q) {
      const auto position = layout.PhysicalPositionOf(q);
      if (position >= zones_.eviction_floor && position < zones_.tile_qubits) {
        seed.Add(q);
      }
    }
    return seed;
  }

  GateBatchPlan BuildCandidatePlan(
      const PendingGateList& gates, const QubitLayout& layout,
      RemapSlotSet& slots) const {
    GateBatchPlan plan(zones_.num_qubits);
    DeferredGates deferred(zones_.num_qubits);

    for (const auto* gate = gates.NextPending(); gate != nullptr;
         gate = gates.NextPending(gate)) {
      if (!deferred.Blocks(*gate) &&
          TryAdmitQubits(gate->Qubits(), layout, slots)) {
        plan.AddGate(*gate);
      } else {
        deferred.Defer(*gate);
      }
    }

    plan.num_swaps = CountSwapsNeeded(plan, layout);
    return plan;
  }

  bool NeedsRemapSlot(unsigned q, const QubitLayout& layout) const {
    return layout.PhysicalPositionOf(q) >= zones_.eviction_floor;
  }

  // Adds the gate's qubits to `slots` if the remap zone has room for them.
  bool TryAdmitQubits(
      const std::vector<unsigned>& qubits, const QubitLayout& layout,
      RemapSlotSet& slots) const {
    unsigned new_slots = 0;
    for (unsigned q : qubits) {
      if (!slots.Contains(q) && NeedsRemapSlot(q, layout)) ++new_slots;
    }
    if (slots.NumSlotsUsed() + new_slots > zones_.NumRemapSlots()) return false;

    for (unsigned q : qubits) {
      if (NeedsRemapSlot(q, layout)) slots.Add(q);
    }
    return true;
  }

  unsigned CountSwapsNeeded(
      const GateBatchPlan& plan, const QubitLayout& layout) const {
    unsigned count = 0;
    for (unsigned q = 0; q < zones_.num_qubits; ++q) {
      if (plan.uses_qubit[q] &&
          layout.PhysicalPositionOf(q) >= zones_.tile_qubits) {
        ++count;
      }
    }
    return count;
  }

  const QubitZones zones_;
  unsigned max_lookahead_seeds_;
};

// --- Running a batch ---

// Sorts the qubits in ascending order, as the simulator expects, and permutes
// the matrix to match.
template <typename FP>
void NormalizeGateQubitOrder(
    std::vector<unsigned>& qubits, Matrix<FP>& matrix) {
  if (qubits.size() < 2) return;
  auto permutation = NormalToGateOrderPermutation(qubits);
  if (!permutation.empty()) {
    MatrixShuffle(permutation, qubits.size(), matrix);
    std::sort(qubits.begin(), qubits.end());
  }
}

// Returns `raw_gate` rewritten to act on the physical positions that `layout`
// stores its logical qubits at, with the qubits sorted as the simulator
// expects.
template <typename FP>
Gate<FP> MakePhysicalGate(
    const Gate<FP>& raw_gate, const QubitLayout& layout, unsigned time) {
  // kind, params and swapped keep their zero defaults; the fuser ignores them.
  Gate<FP> physical_gate{};
  physical_gate.time = time;
  for (unsigned q : raw_gate.qubits) {
    physical_gate.qubits.push_back(layout.PhysicalPositionOf(q));
  }
  physical_gate.matrix = raw_gate.matrix;
  NormalizeGateQubitOrder(physical_gate.qubits, physical_gate.matrix);
  return physical_gate;
}

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
  using SimulationStats = gate_batch_internal::SimulationStats;
  using ExecutableGate = FusedGate<fp_type>;
  using GateBatchPlanner = gate_batch_internal::GateBatchPlanner<fp_type>;
  using GateBatchPlan = gate_batch_internal::GateBatchPlan;
  using QubitZones = gate_batch_internal::QubitZones;

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
        layout_(layout),
        gate_batch_planner_(zones_, param.max_lookahead_seeds) {
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

  // --- Simulation flow ---

  template <typename Circuit>
  bool SimulateCircuit(const Circuit& circuit, QubitOrderOnExit order) {
    if (!ValidateParameters() || !backend_.Prepare()) return false;
    simulation_stats_.num_raw_gates = unsigned(circuit.ops.size());

    if (zones_.IsSingleTile()) return SimulateSingleTile(circuit);
    return SimulateMultiTile(circuit, order);
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

  // The whole state is one tile: no planning or swaps, just fuse the circuit
  // and run it once.
  template <typename Circuit>
  bool SimulateSingleTile(const Circuit& circuit) {
    const bool fused = TimeHostWork(simulation_stats_.fuse_seconds, [&] {
      return FuseBatchGates(circuit.ops, executable_gates_);
    });
    if (!fused) return false;

    const auto simulation_start = StartSimulationClock();
    ExecuteFusedGates(executable_gates_);
    simulation_stats_.num_gate_batches = circuit.ops.empty() ? 0 : 1;
    FinishSimulation(simulation_start);
    return true;
  }

  template <typename Circuit>
  bool SimulateMultiTile(const Circuit& circuit, QubitOrderOnExit order) {
    using Op = typename std::decay_t<decltype(circuit.ops)>::value_type;

    const bool prepared = TimeHostWork(simulation_stats_.prepare_seconds, [&] {
      return PreparePendingGates(circuit);
    });
    if (!prepared) return false;

    const auto simulation_start = StartSimulationClock();
    FillHotZone();

    std::vector<Op> batch_operations;

    while (pending_gates_.NextPending() != nullptr) {
      if (!ExecuteNextGateBatch(batch_operations)) return false;
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
  bool ExecuteNextGateBatch(std::vector<Op>& batch_operations) {
    const auto plan = TimeHostWork(simulation_stats_.plan_seconds, [&] {
      return gate_batch_planner_.PlanNextGateBatch(pending_gates_, layout_);
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

    const bool fused = TimeHostWork(simulation_stats_.fuse_seconds, [&] {
      BuildBatchOperations(plan, batch_operations);
      return FuseBatchGates(batch_operations, executable_gates_);
    });
    if (!fused) return false;
    pending_gates_.MarkApplied(plan.gate_indices);

    ExecuteFusedGates(executable_gates_);
    ++simulation_stats_.num_gate_batches;
    return true;
  }

  // --- Gate preparation ---

  static void ReportUnsupportedOperation() {
    IO::errorf(
        "qsim_gate_batch: only plain gates are supported, not controlled "
        "gates or measurements; use QSimRunner for those.\n");
  }

  template <typename Circuit>
  bool PreparePendingGates(const Circuit& circuit) {
    auto pending_gates = PendingGateList::FromCircuit(circuit);
    if (!pending_gates) {
      ReportUnsupportedOperation();
      return false;
    }
    pending_gates_ = std::move(*pending_gates);
    return true;
  }

  // --- Hot-zone placement ---

  std::vector<uint64_t> ComputeQubitUsageScores() const {
    std::vector<uint64_t> scores(zones_.num_qubits, 0);
    for (std::size_t idx = 0; idx < pending_gates_.Size(); ++idx) {
      const auto& qubits = pending_gates_[idx].Qubits();
      for (unsigned q : qubits) scores[q] += qubits.size();
    }
    return scores;
  }

  static std::vector<char> SelectHighestScoringQubits(
      const std::vector<uint64_t>& scores, std::vector<unsigned> candidates,
      unsigned count) {
    assert(count <= candidates.size());
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [&scores](unsigned a, unsigned b) { return scores[a] > scores[b]; });

    std::vector<char> selected(scores.size(), 0);
    for (unsigned i = 0; i < count; ++i) selected[candidates[i]] = 1;
    return selected;
  }

  void FillHotZone() {
    if (zones_.NumHotSlots() == 0) return;

    // Lane qubits never move, so only qubits stored above the lanes can be
    // made hot.
    std::vector<unsigned> candidates;
    for (unsigned q = 0; q < zones_.num_qubits; ++q) {
      if (layout_.PhysicalPositionOf(q) >= zones_.lane_qubits) {
        candidates.push_back(q);
      }
    }

    const auto usage_scores = ComputeQubitUsageScores();
    const auto is_hot = SelectHighestScoringQubits(
        usage_scores, std::move(candidates), zones_.NumHotSlots());

    const auto num_swaps =
        MoveQubitsIntoZone(is_hot, zones_.lane_qubits, zones_.eviction_floor);
    LogHotZonePlacement(usage_scores, num_swaps);
  }

  // --- Qubit swaps ---

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
    auto& swaps = swap_pairs_;
    swaps.clear();
    auto evict_position = zone_end;

    for (unsigned q = 0; q < wanted.size(); ++q) {
      if (!wanted[q]) continue;
      const auto position = layout_.PhysicalPositionOf(q);
      if (position < zone_end) continue;

      do {
        assert(evict_position > zone_begin);
        --evict_position;
      } while (wanted[layout_.LogicalQubitAt(evict_position)]);
      swaps.emplace_back(evict_position, position);
      layout_.SwapPhysicalPositions(evict_position, position);
    }

    ApplySwapsToState(swaps);
    return swaps.size();
  }

  void ApplySwapsToState(const std::vector<QubitSwap>& swap_pairs) {
    if (swap_pairs.empty()) return;
    TimeBackendWork(simulation_stats_.swap_seconds, [&] {
      backend_.ApplySwaps(swap_pairs);
    });
    simulation_stats_.num_swaps += swap_pairs.size();
  }

  void RestoreIdentityQubitOrder() {
    while (layout_.BuildIdentityRestorationSwaps(swap_pairs_)) {
      ApplySwapsToState(swap_pairs_);
      ++simulation_stats_.num_restore_passes;
    }
  }

  // --- Batch building, fusion and execution ---

  template <typename Op>
  void BuildBatchOperations(
      const GateBatchPlan& plan, std::vector<Op>& batch_operations) const {
    batch_operations.clear();
    batch_operations.reserve(plan.NumGates());

    unsigned time = 0;
    for (std::size_t idx : plan.gate_indices) {
      batch_operations.push_back(
          Op{gate_batch_internal::MakePhysicalGate(
              *pending_gates_[idx].gate, layout_, time++)});
    }
  }

  // Fuses `batch_operations` into `executable_gates`, replacing its contents.
  template <typename Op>
  bool FuseBatchGates(
      const std::vector<Op>& batch_operations,
      std::vector<ExecutableGate>& executable_gates) {
    auto fused_ops =
        Fuser::FuseGates(param_, zones_.tile_qubits, batch_operations);
    if (fused_ops.empty() && !batch_operations.empty()) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return false;
    }

    executable_gates.clear();
    executable_gates.reserve(fused_ops.size());
    // Fused ops reference input batch_operations, so copying must happen while
    // it is alive.
    for (auto& operation : fused_ops) {
      auto* gate = OpGetAlternative<ExecutableGate>(operation);
      if (gate == nullptr) {
        ReportUnsupportedOperation();
        return false;
      }
      if (gate->qubits.size() > Backend::kMaxGateQubits) {
        IO::errorf("qsim_gate_batch: gate wider than the backend supports.\n");
        return false;
      }
      assert(std::is_sorted(gate->qubits.begin(), gate->qubits.end()));
      executable_gates.push_back(std::move(*gate));
    }
    simulation_stats_.num_executable_gates += executable_gates.size();
    return true;
  }

  void ExecuteFusedGates(const std::vector<ExecutableGate>& executable_gates) {
    TimeBackendWork(simulation_stats_.gate_seconds, [&] {
      backend_.ExecuteBatch(executable_gates);
    });
  }

  // --- Timing and logging ---

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
  PendingGateList pending_gates_;
  QubitLayout& layout_;
  GateBatchPlanner gate_batch_planner_;

  // Reused across batches so the batch loop does not reallocate.
  std::vector<QubitSwap> swap_pairs_;
  std::vector<ExecutableGate> executable_gates_;

  // Accumulated over the whole run; printed by LogSimulationSummary.
  SimulationStats simulation_stats_;
};

}  // namespace qsim

#endif  // GATE_BATCH_RUNNER_H_
