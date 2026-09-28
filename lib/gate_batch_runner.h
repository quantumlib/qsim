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

// Gate-batched cache-local simulation runner, independent of where the state
// lives. Follows the original proposal as closely as possible (see
// original_proposal.md).
//
// Data hierarchy:
//   Circuit: the full program, as an ordered list of raw gates.
//   Tile: a fast-memory-sized slice of 2^tile_qubits amplitudes (held in a
//         CPU cache); one gate batch runs against every tile.
//   Lane group: the vectorized unit of the state layout - 2^lane_qubits
//               complex amplitudes, one per SIMD lane (NEON/SSE: 4, AVX2: 8,
//               AVX512: 16).
//
// Algorithm:
//   Move the most-used logical qubits into the fixed low zone up front
//   (PlaceHotQubitsInFixedZone), at the cost of one swap pass; they then
//   stay resident for every gate batch.
//
//   while gates remain:
//     Select the next gate batch (GateBatchPlanner::PlanNextGateBatch):
//       score a bounded set of candidate qubit sets - the greedy
//       first-fit scan, the qubits already resident, and a few sets
//       seeded from upcoming gates - and keep the best.
//     Remap the batch's qubits into physical positions [0, tile_qubits)
//       with one pass of disjoint transpositions (BuildSwapsBelow,
//       Backend::ApplySwaps). The backend's pinned lane positions are never
//       swapped; every other qubit draws from a bounded remap budget
//       (GateBatchPlanner::RemapSlotCapacity).
//     Fuse the batch's gates on physical qubits with the standard fuser
//       (FuseBatchGates).
//     Execute every fused gate on every tile (Backend::ExecuteBatch).
//
//   Restore identity qubit order with a final sequence of disjoint-
//   transposition passes (RestoreIdentityQubitOrder).
//
// Gates are scheduled raw: there is no global pre-fusion, 1q-chain
// peephole, or diagonal-gate phase kernel/bundle. All fusion happens
// inside the per-batch fuser above.
//
// Backends (run_qsim_gate_batch.h for CPUs) own the state memory and
// provide:
//   using State, StateSpace, fp_type;
//   struct Parameter;  // backend-specific options, part of the runner's
//   static constexpr unsigned kDefaultTileQubits, kDefaultEvictionFloor;
//   Backend(const RunnerParameter&, unsigned num_qubits, State&);
//   unsigned TileQubits() const;   // effective tile size
//   unsigned LaneQubits() const;   // low positions the planner never remaps
//   bool Prepare();                // validate options, set up threads
//   void ApplySwaps(const std::vector<QubitSwap>&);
//   void ExecuteBatch(const std::vector<ExecutableGate<fp_type>>&);
//   void Synchronize();            // wait for queued work (timing, finish)

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

// ======== Plain records ========

// Internal scheduling representation of one raw circuit gate.
template <typename FP>
struct PendingGate {
  std::size_t Arity() const { return logical_qubits.size(); }

  std::vector<unsigned> logical_qubits;  // ascending
  Matrix<FP> matrix;

  // Diagonal gates commute with each other, so an overlapping pair of them
  // imposes no ordering constraint on the planner.
  bool is_diagonal = false;

  // Set once the gate has executed, and only after its batch has fully
  // succeeded. Planning scans skip these.
  bool applied = false;
};

// A fused gate ready to execute inside a tile.
template <typename FP>
struct ExecutableGate {
  std::vector<unsigned> physical_qubits;  // ascending
  Matrix<FP> matrix;
};

// Buffers reused by every gate batch.
template <typename FP>
struct GateBatchWorkspace {
  std::vector<QubitSwap> swap_pairs;
  std::vector<ExecutableGate<FP>> executable_gates;
};

// Counters and step timings accumulated across gate batches. Gate batches
// and identity-restoration passes are counted separately because only gate
// batches make circuit progress; restore passes are pure overhead. Timings
// are always collected but reported only at verbosity > 1.
struct SimulationStats {
  unsigned num_gate_batches = 0;
  unsigned num_restore_passes = 0;
  unsigned num_swaps = 0;
  unsigned num_executable_gates = 0;
  double plan_seconds = 0.0;
  double swap_seconds = 0.0;
  double fuse_seconds = 0.0;
  double gate_seconds = 0.0;
};

// ======== Gate-batch planning ========

// Logical qubits selected for one gate batch, with eviction-aware remap-slot
// accounting. Logical qubits resident below the eviction floor join for free.
struct GateBatchQubitSet {
  bool Contains(unsigned logical_qubit) const {
    return contains_qubit[logical_qubit];
  }

  std::vector<char> contains_qubit;
  unsigned remap_slots_used = 0;
};

// The outcome of planning one gate batch, built without mutating gates,
// layout, or state.
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

// Chooses the gates for the next gate batch. First-fit selection is greedy
// maximal, not maximum: an early gate can fill the qubit set and shut out
// a larger family of later gates. Rather than solve that exactly (it is
// combinatorial), PlanNextGateBatch tries a bounded set of candidate
// qubit sets and keeps the best-scoring plan:
//   - the first-fit greedy set (baseline: never worse than a single
//     greedy scan, because re-evaluating a finished set can only
//     admit more gates);
//   - the currently resident set (a zero-swap batch);
//   - a set seeded by each of the next few pending gates, for when
//     the first pending gate is the one poisoning the set.
// PlanNextGateBatch reads gates and layout but never modifies them.
template <typename FP>
class GateBatchPlanner {
 public:
  GateBatchPlanner(unsigned num_state_qubits, unsigned tile_qubits,
                   unsigned lane_qubits, unsigned min_eviction_floor,
                   unsigned max_gate_seeds, bool commute_diagonal_gates)
      : num_logical_qubits_(num_state_qubits),
        tile_qubits_(tile_qubits),
        eviction_floor_(ComputeEvictionFloor(num_state_qubits, tile_qubits,
                                             lane_qubits,
                                             min_eviction_floor)),
        max_gate_seeds_(max_gate_seeds),
        commute_diagonal_gates_(commute_diagonal_gates),
        is_qubit_blocked_(num_state_qubits),
        is_qubit_blocked_for_diagonal_(num_state_qubits) {}

  GateBatchPlan PlanNextGateBatch(const std::vector<PendingGate<FP>>& gates,
                                  const QubitLayout& layout) {
    GateBatchPlan best_plan;
    // Ties keep the first candidate, so priority follows evaluation order:
    // greedy, resident, then seeded plans in circuit order.
    auto consider = [&](GateBatchQubitSet seed) {
      GrowQubitSetGreedily(gates, layout, seed);
      auto plan = EvaluateQubitSet(gates, layout, seed);
      if (!best_plan.HasGates() || plan.score > best_plan.score) {
        best_plan = std::move(plan);
      }
    };

    // First-fit greedy, then the zero-swap resident set.
    consider(MakeEmptyQubitSet());
    consider(MakeResidentQubitSet(layout));

    // One candidate per following pending gate, for when the first pending
    // gate is the one poisoning the set.
    unsigned seeds_used = 0;
    bool skipped_first_pending = false;
    for (const PendingGate<FP>& gate : gates) {
      if (gate.applied) continue;
      if (!skipped_first_pending) {
        skipped_first_pending = true;
        continue;
      }
      if (seeds_used++ == max_gate_seeds_) break;
      auto seed = MakeEmptyQubitSet();
      if (TryAdmitQubits(gate.logical_qubits, layout, seed)) {
        consider(std::move(seed));
      }
    }
    return best_plan;
  }

  unsigned EvictionFloor() const { return eviction_floor_; }

 private:
  // Keep enough remap positions for an ordinary two-qubit gate even when
  // the tile is too small to reach min_eviction_floor.
  static constexpr unsigned kMinRemapSlots = 2;

  // Starting any remapping incurs a full-state pass. Additional pairs share
  // that pass and therefore carry a smaller marginal cost. With these
  // weights, eight pairs offset one additional gate.
  static constexpr double kSwapPassCost = 0.5;
  static constexpr double kSwapPairCost = 1.0 / 16.0;

  GateBatchQubitSet MakeEmptyQubitSet() const {
    GateBatchQubitSet qubit_set;
    qubit_set.contains_qubit.assign(num_logical_qubits_, 0);
    return qubit_set;
  }

  unsigned RemapSlotCapacity() const {
    return tile_qubits_ - eviction_floor_;
  }

  // Positions below the eviction floor are fixed residents and join a set
  // for free. Positions at or above it consume one remap slot: a low
  // resident protects a potential victim, while a high qubit requires a
  // victim. Keeping the floor at min_eviction_floor makes every remap span
  // at least 2^(floor + 3) bytes (256 B at floor 5), whatever the lane width.
  // Small tiles retain at least two remap slots so an ordinary two-qubit
  // gate can make progress.
  static unsigned ComputeEvictionFloor(unsigned num_logical_qubits,
                                       unsigned tile_qubits,
                                       unsigned lane_qubits,
                                       unsigned min_eviction_floor) {
    if (tile_qubits == num_logical_qubits) return tile_qubits;

    const auto full =
        tile_qubits > lane_qubits ? tile_qubits - lane_qubits : 0u;
    const auto depth_capped = tile_qubits > min_eviction_floor
                                  ? tile_qubits - min_eviction_floor
                                  : 0u;
    const auto capacity =
        std::min(full, std::max(depth_capped, kMinRemapSlots));
    return tile_qubits - capacity;
  }

  // The zero-swap candidate: exactly the qubits already resident in
  // the low tile. Its occupancy may exceed the depth-capped capacity;
  // that is safe because none of these qubits needs a swap (no
  // evictions happen), and growth beyond them is still capacity-bound.
  GateBatchQubitSet MakeResidentQubitSet(
      const QubitLayout& layout) const {
    auto qubit_set = MakeEmptyQubitSet();
    for (unsigned q = 0; q < num_logical_qubits_; ++q) {
      if (layout.PhysicalPositionOf(q) < tile_qubits_) {
        qubit_set.remap_slots_used +=
            AdditionalRemapSlotsFor(q, qubit_set, layout);
        qubit_set.contains_qubit[q] = 1;
      }
    }
    return qubit_set;
  }

  // First-fit growth: scan pending gates in circuit order, admitting
  // each gate's qubits when they fit and blocking them otherwise (the
  // same causal rule EvaluateQubitSet applies to the finished set).
  void GrowQubitSetGreedily(const std::vector<PendingGate<FP>>& gates,
                            const QubitLayout& layout,
                            GateBatchQubitSet& qubit_set) {
    ClearBlockedQubits();
    for (const PendingGate<FP>& gate : gates) {
      if (gate.applied) continue;
      if (AnyQubitBlocked(gate) ||
          !TryAdmitQubits(gate.logical_qubits, layout, qubit_set)) {
        BlockQubits(gate);
      }
    }
  }

  // Exact evaluation of a fixed qubit set: one causal-blocking scan over
  // the pending gates collects every gate the set can execute, then the
  // plan is scored.
  GateBatchPlan EvaluateQubitSet(
      const std::vector<PendingGate<FP>>& gates, const QubitLayout& layout,
      const GateBatchQubitSet& qubit_set) {
    GateBatchPlan plan;
    plan.uses_qubit.assign(num_logical_qubits_, 0);
    ClearBlockedQubits();

    for (std::size_t idx = 0; idx < gates.size(); ++idx) {
      const auto& gate = gates[idx];
      if (gate.applied) continue;
      if (!AnyQubitBlocked(gate) &&
          QubitSetContainsAll(gate.logical_qubits, qubit_set)) {
        plan.gate_indices.push_back(idx);
        for (unsigned q : gate.logical_qubits) {
          plan.uses_qubit[q] = 1;
        }
      } else {
        BlockQubits(gate);
      }
    }

    const auto required_swaps = CountSwapsNeeded(plan, layout);
    const auto swap_cost =
        required_swaps == 0 ? 0.0
                            : kSwapPassCost + kSwapPairCost * required_swaps;
    plan.score = plan.NumGates() - swap_cost;
    return plan;
  }

  // Remap-slot cost of admitting q. Fixed residents below the eviction
  // floor cost nothing. A resident in the eviction zone consumes a victim
  // position by protecting it, and a high qubit consumes one by entering.
  unsigned AdditionalRemapSlotsFor(
      unsigned q, const GateBatchQubitSet& qubit_set,
      const QubitLayout& layout) const {
    if (qubit_set.Contains(q)) return 0;
    return layout.PhysicalPositionOf(q) < eviction_floor_ ? 0 : 1;
  }

  // Admits all of a gate's qubits into the set, or none of them.
  bool TryAdmitQubits(const std::vector<unsigned>& qubits,
                      const QubitLayout& layout,
                      GateBatchQubitSet& qubit_set) const {
    unsigned additional_remap_slots = 0;
    for (unsigned q : qubits) {
      additional_remap_slots +=
          AdditionalRemapSlotsFor(q, qubit_set, layout);
    }
    if (qubit_set.remap_slots_used + additional_remap_slots >
        RemapSlotCapacity()) {
      return false;
    }
    for (unsigned q : qubits) qubit_set.contains_qubit[q] = 1;
    qubit_set.remap_slots_used += additional_remap_slots;
    return true;
  }

  static bool QubitSetContainsAll(const std::vector<unsigned>& qubits,
                                  const GateBatchQubitSet& qubit_set) {
    return std::all_of(
        qubits.begin(), qubits.end(),
        [&qubit_set](unsigned q) { return qubit_set.Contains(q); });
  }

  unsigned CountSwapsNeeded(const GateBatchPlan& plan,
                            const QubitLayout& layout) const {
    unsigned count = 0;
    for (unsigned q = 0; q < num_logical_qubits_; ++q) {
      if (plan.UsesQubit(q) &&
          layout.PhysicalPositionOf(q) >= tile_qubits_) {
        ++count;
      }
    }
    return count;
  }

  void ClearBlockedQubits() {
    std::fill(is_qubit_blocked_.begin(), is_qubit_blocked_.end(), 0);
    std::fill(is_qubit_blocked_for_diagonal_.begin(),
              is_qubit_blocked_for_diagonal_.end(), 0);
  }

  bool TreatAsCommuting(const PendingGate<FP>& gate) const {
    return commute_diagonal_gates_ && gate.is_diagonal;
  }

  // A skipped non-diagonal gate is a hard barrier for everything. A skipped
  // diagonal gate only bars later non-diagonal gates, since it can be
  // reordered past any other diagonal gate.
  bool AnyQubitBlocked(const PendingGate<FP>& gate) const {
    const auto commuting = TreatAsCommuting(gate);
    for (unsigned q : gate.logical_qubits) {
      if (is_qubit_blocked_[q]) return true;
      if (!commuting && is_qubit_blocked_for_diagonal_[q]) return true;
    }
    return false;
  }

  void BlockQubits(const PendingGate<FP>& gate) {
    const auto commuting = TreatAsCommuting(gate);
    for (unsigned q : gate.logical_qubits) {
      if (!commuting) is_qubit_blocked_[q] = 1;
      is_qubit_blocked_for_diagonal_[q] = 1;
    }
  }

  unsigned num_logical_qubits_;
  unsigned tile_qubits_;
  unsigned eviction_floor_;
  unsigned max_gate_seeds_;
  bool commute_diagonal_gates_;
  std::vector<char> is_qubit_blocked_;
  std::vector<char> is_qubit_blocked_for_diagonal_;
};

}  // namespace gate_batch_internal

template <typename IO, typename Fuser, typename Backend>
class GateBatchRunner {
 public:
  using StateSpace = typename Backend::StateSpace;
  using State = typename Backend::State;
  using QubitMappedState = qsim::QubitMappedState<State>;
  using fp_type = typename Backend::fp_type;
  static_assert(std::is_same_v<fp_type, float>,
                "GateBatchRunner requires a float state space.");

  struct Parameter : public Fuser::Parameter, public Backend::Parameter {
    Parameter() { this->max_fused_size = 3; }

    unsigned tile_qubits = Backend::kDefaultTileQubits;

    // Lowest eviction position allowed in a multi-tile gate batch. Swap-pass
    // spans are 2^(floor - lane_qubits) lane groups, so a higher floor keeps
    // spans longer; lowering it widens the remap budget.
    unsigned min_eviction_floor = Backend::kDefaultEvictionFloor;

    // Seed the fixed zone [lane_qubits, eviction_floor) with the most-used
    // logical qubits before the first gate batch, at the cost of one swap
    // pass over the state.
    bool place_hot_qubits = true;

    // How many pending gates beyond the first get to seed their own candidate
    // qubit set. Planning is a negligible fraction of runtime, so this buys
    // batch size almost for free.
    unsigned max_gate_seeds = 64;

    // Treat an overlap between two diagonal gates as non-blocking. Exact, not
    // an approximation: diagonal matrices commute.
    bool commute_diagonal_gates = false;
  };

  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  State& state) {
    QubitLayout layout(circuit.num_qubits);
    GateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuit(circuit, true);
  }

  // Runs directly on a mapped state and leaves its physical storage in the
  // returned logical-to-physical layout, avoiding final restoration.
  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  QubitMappedState& state) {
    GateBatchRunner runner(param, circuit.num_qubits, state.state,
                           state.layout);
    return runner.SimulateCircuit(circuit, false);
  }

 private:
  using PendingGate = gate_batch_internal::PendingGate<fp_type>;
  using ExecutableGate = gate_batch_internal::ExecutableGate<fp_type>;
  using SimulationStats = gate_batch_internal::SimulationStats;
  using GateBatchWorkspace = gate_batch_internal::GateBatchWorkspace<fp_type>;
  using GateBatchPlanner = gate_batch_internal::GateBatchPlanner<fp_type>;
  using GateBatchPlan = gate_batch_internal::GateBatchPlan;

  GateBatchRunner(const Parameter& param, unsigned num_qubits, State& state,
                  QubitLayout& layout)
      : param_(param),
        num_qubits_(num_qubits),
        backend_(param, num_qubits, state),
        lane_qubits_(backend_.LaneQubits()),
        layout_(layout),
        gate_batch_planner_(num_qubits, backend_.TileQubits(), lane_qubits_,
                            param.min_eviction_floor, param.max_gate_seeds,
                            param.commute_diagonal_gates) {
    assert(layout_.NumQubits() == num_qubits_);
  }

  // ======== Drivers ========

  template <typename Circuit>
  bool SimulateCircuit(const Circuit& circuit, bool restore_qubit_order) {
    using Op = typename std::decay_t<decltype(circuit.ops)>::value_type;

    if (param_.max_fused_size < 2) {
      IO::errorf("qsim_gate_batch: max_fused_size must be at least 2.\n");
      return false;
    }
    if (!backend_.Prepare()) return false;
    const auto prepare_start = GetTime();
    if (!PreparePendingGates(circuit)) return false;
    LogPreparationTime(prepare_start);

    backend_.Synchronize();
    const auto simulation_start = GetTime();
    PlaceHotQubitsInFixedZone();

    // Op depends on Circuit, so this buffer remains local rather than
    // forcing the whole runner type to acquire another template parameter.
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

  // Executes one complete gate batch following the proposal's loop body.
  // Returns the number of gates applied; 0 signals failure (fuser error, or
  // a schedule that cannot make progress). Planning is read-only; gates are
  // marked applied only after fusion succeeds, so a failed batch never
  // corrupts the schedule.
  template <typename Op>
  std::size_t ExecuteNextGateBatch(std::vector<Op>& batch_operations) {
    const auto tile_qubits = backend_.TileQubits();

    // pick_maximum_number_of_gates_acting_on_low_qubits
    const auto plan_start = GetTime();
    const auto plan = gate_batch_planner_.PlanNextGateBatch(
        pending_gates_, layout_);
    simulation_stats_.plan_seconds += GetTime() - plan_start;
    if (!plan.HasGates()) {
      IO::errorf("qsim_gate_batch: a gate does not fit the qubit set "
                 "of %u tile qubits; use a larger tile_qubits.\n",
                 tile_qubits);
      return 0;
    }

    // select_qubits_for_swap + swap_low_and_high_qubits
    BuildSwapsBelow(plan.uses_qubit, tile_qubits,
                    gate_batch_workspace_.swap_pairs);
    ApplySwapsToState(gate_batch_workspace_.swap_pairs);

    // fused_low_gates = fuse(low_gates)
    const auto fuse_start = GetTime();
    BuildBatchOperations(plan, batch_operations);
    if (!FuseBatchGates(batch_operations)) {
      return 0;
    }
    simulation_stats_.fuse_seconds += GetTime() - fuse_start;
    simulation_stats_.num_executable_gates +=
        gate_batch_workspace_.executable_gates.size();
    MarkGatesApplied(plan);

    // for i in 0..(2^num_high_qubits): apply all fused gates to tile i
    const auto gates_start = StartTimer();
    backend_.ExecuteBatch(gate_batch_workspace_.executable_gates);
    simulation_stats_.gate_seconds += StopTimer(gates_start);

    ++simulation_stats_.num_gate_batches;
    return plan.NumGates();
  }

  // ======== Step 1: pending-gate preparation ========

  // Sorts a gate's qubits ascending, permuting `matrix` to match.
  static void NormalizeGateQubitOrder(std::vector<unsigned>& qubits,
                                      Matrix<fp_type>& matrix) {
    if (qubits.size() < 2) return;
    auto permutation = NormalToGateOrderPermutation(qubits);
    if (!permutation.empty()) {
      MatrixShuffle(permutation, qubits.size(), matrix);
      std::sort(qubits.begin(), qubits.end());
    }
  }

  // Checked on the matrix rather than the gate kind so parameterized and
  // already-fused diagonal gates are recognized too.
  static bool IsDiagonalMatrix(const Matrix<fp_type>& matrix,
                               std::size_t arity) {
    const auto dim = std::size_t{1} << arity;
    // Row-major complex entries, stored as (real, imaginary) float pairs.
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

  // The proposal starts from the raw circuit: every gate becomes one
  // pending gate, verbatim (qubit order normalized). All fusion is deferred
  // to the per-batch fuser. Controlled gates, measurements, and channels
  // are not supported.
  template <typename Circuit>
  bool PreparePendingGates(const Circuit& circuit) {
    pending_gates_.reserve(circuit.ops.size());

    for (const auto& operation : circuit.ops) {
      const auto* raw_gate =
          OpGetAlternative<Gate<fp_type>>(operation);
      if (raw_gate == nullptr) {
        IO::errorf("qsim_gate_batch: unsupported operation "
                   "(controlled gate or measurement).\n");
        return false;
      }

      auto logical_qubits = raw_gate->qubits;
      auto matrix = raw_gate->matrix;
      NormalizeGateQubitOrder(logical_qubits, matrix);
      const bool diagonal =
          IsDiagonalMatrix(matrix, logical_qubits.size());
      pending_gates_.push_back(
          {std::move(logical_qubits), std::move(matrix), diagonal});
    }
    return true;
  }

  // ======== Step 2: hot-qubit placement ========

  // Scores qubits by gate participation, weighted by gate arity.
  std::vector<uint64_t> ComputeQubitUsageScores() const {
    std::vector<uint64_t> scores(num_qubits_, 0);
    for (const PendingGate& gate : pending_gates_) {
      const auto weight = gate.Arity();
      for (unsigned q : gate.logical_qubits) scores[q] += weight;
    }
    return scores;
  }

  // Marks the count highest-scoring qubits at or above first_qubit. Among
  // equal scores, the lower qubit index wins.
  static std::vector<char> SelectHighestScoringQubits(
      const std::vector<uint64_t>& scores, unsigned first_qubit,
      unsigned count) {
    std::vector<unsigned> candidates(scores.size() - first_qubit);
    std::iota(candidates.begin(), candidates.end(), first_qubit);
    assert(count <= candidates.size());
    // Stable, so equal scores keep ascending qubit order.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&scores](unsigned a, unsigned b) {
                       return scores[a] > scores[b];
                     });

    std::vector<char> selected(scores.size(), 0);
    for (unsigned i = 0; i < count; ++i) selected[candidates[i]] = 1;
    return selected;
  }

  // Places the most frequently used logical qubits in the fixed zone
  // [lane_qubits, eviction_floor), just above the lane positions,
  // which remain untouched. Canonical runs restore qubit order at the end.
  void PlaceHotQubitsInFixedZone() {
    const auto eviction_floor = gate_batch_planner_.EvictionFloor();
    if (!param_.place_hot_qubits || backend_.TileQubits() == num_qubits_ ||
        eviction_floor <= lane_qubits_) {
      return;
    }

    const auto usage_scores = ComputeQubitUsageScores();
    const auto is_hot = SelectHighestScoringQubits(
        usage_scores, lane_qubits_, eviction_floor - lane_qubits_);

    auto& swap_pairs = gate_batch_workspace_.swap_pairs;
    BuildSwapsBelow(is_hot, eviction_floor, swap_pairs);
    LogFixedZonePlacement(usage_scores, eviction_floor, swap_pairs.size());
    ApplySwapsToState(swap_pairs);
  }

  // ======== Step 3: qubit remapping ========

  // Moves every wanted logical qubit into a physical position below
  // `limit`, recording the transpositions in `swaps` and updating the
  // layout to match. Victims are taken from limit-1 downward, skipping
  // positions that already hold a wanted qubit; the caller guarantees
  // enough victims above the pinned lane positions.
  void BuildSwapsBelow(const std::vector<char>& is_wanted, unsigned limit,
                       std::vector<QubitSwap>& swaps) {
    swaps.clear();
    auto victim = limit;

    for (unsigned q = 0; q < is_wanted.size(); ++q) {
      if (!is_wanted[q]) continue;
      const auto position = layout_.PhysicalPositionOf(q);
      if (position < limit) continue;

      do {
        assert(victim > lane_qubits_);
        --victim;
      } while (is_wanted[layout_.LogicalQubitAt(victim)]);
      swaps.emplace_back(victim, position);
      layout_.SwapPhysicalPositions(victim, position);
    }
  }

  // Applies the transpositions to the state in one pass.
  void ApplySwapsToState(const std::vector<QubitSwap>& swap_pairs) {
    if (swap_pairs.empty()) return;
    const auto swap_start = StartTimer();
    backend_.ApplySwaps(swap_pairs);
    simulation_stats_.swap_seconds += StopTimer(swap_start);
    simulation_stats_.num_swaps += swap_pairs.size();
  }

  // ======== Step 4: fusion ========

  // Rebuilds the plan's pending gates as ordinary gates on PHYSICAL
  // qubits, with fresh sequential times, so the standard fuser can
  // consume them as if they were a small L-qubit circuit.
  template <typename Op>
  void BuildBatchOperations(const GateBatchPlan& plan,
                            std::vector<Op>& batch_operations) const {
    batch_operations.clear();
    batch_operations.reserve(plan.NumGates());

    unsigned time = 0;
    for (std::size_t idx : plan.gate_indices) {
      const auto& gate = pending_gates_[idx];

      std::vector<unsigned> physical_qubits(gate.Arity());
      for (std::size_t i = 0; i < gate.Arity(); ++i) {
        physical_qubits[i] =
            layout_.PhysicalPositionOf(gate.logical_qubits[i]);
      }
      auto physical_matrix = gate.matrix;
      NormalizeGateQubitOrder(physical_qubits, physical_matrix);

      // Synthetic gate: kind is irrelevant outside qsimh; sequential
      // times satisfy the fuser's ordering contract.
      auto physical_gate = Gate<fp_type>{
          {0, time++, std::move(physical_qubits)}, {},
          std::move(physical_matrix), false};
      batch_operations.push_back(Op{std::move(physical_gate)});
    }
  }

  // fuse(low_gates): runs the standard fuser over the batch's physical
  // gates (max_fused_size from param; the proposal suggests 2 or 3) and
  // copies the fused gates into ExecutableGate records. The copy must happen
  // while the fuser's input container is still alive (fused ops point into
  // it). The fuser sorts each fused gate's qubits.
  template <typename Op>
  bool FuseBatchGates(const std::vector<Op>& batch_operations) {
    const auto fused_ops = Fuser::FuseGates(param_, backend_.TileQubits(),
                                            batch_operations);
    if (fused_ops.empty() && !batch_operations.empty()) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return false;
    }

    auto& executable_gates = gate_batch_workspace_.executable_gates;
    executable_gates.clear();
    executable_gates.reserve(fused_ops.size());
    for (const auto& operation : fused_ops) {
      const auto* gate = OpGetAlternative<FusedGate<fp_type>>(operation);
      if (gate == nullptr) {
        IO::errorf("qsim_gate_batch: unsupported operation in gate batch.\n");
        return false;
      }
      assert(std::is_sorted(gate->qubits.begin(), gate->qubits.end()));
      executable_gates.push_back({gate->qubits, gate->matrix});
    }
    return true;
  }

  // Deferred until fusion has succeeded, so a failed batch leaves the
  // schedule untouched.
  void MarkGatesApplied(const GateBatchPlan& plan) {
    for (std::size_t idx : plan.gate_indices) {
      pending_gates_[idx].applied = true;
    }
  }

  // ======== Step 5: cleanup ========

  void RestoreIdentityQubitOrder() {
    std::vector<QubitSwap> swap_pairs;

    while (layout_.BuildIdentityRestorationSwaps(swap_pairs)) {
      ApplySwapsToState(swap_pairs);
      ++simulation_stats_.num_restore_passes;
    }
  }

  // ======== Timing ========

  // Backends may queue work asynchronously, so per-step timings,
  // which are reported only at verbosity > 1, wait for it to finish.
  double StartTimer() {
    if (param_.verbosity > 1) backend_.Synchronize();
    return GetTime();
  }

  double StopTimer(double start) {
    if (param_.verbosity > 1) backend_.Synchronize();
    return GetTime() - start;
  }

  // ======== Diagnostics ========

  void LogPreparationTime(double prepare_start) const {
    if (param_.verbosity <= 1) return;
    IO::messagef("prepare time is %g seconds.\n",
                 GetTime() - prepare_start);
  }

  void LogFixedZonePlacement(const std::vector<uint64_t>& usage_scores,
                             unsigned eviction_floor,
                             std::size_t num_initial_swaps) const {
    if (param_.verbosity <= 1) return;

    IO::messagef("fixed hot zone [%u,%u):", lane_qubits_, eviction_floor);
    for (unsigned p = lane_qubits_; p < eviction_floor; ++p) {
      const auto q = layout_.LogicalQubitAt(p);
      IO::messagef(" q%u(%llu)", q,
                   static_cast<unsigned long long>(usage_scores[q]));
    }
    IO::messagef("; %u initial swaps.\n", unsigned(num_initial_swaps));
  }

  void LogSimulationSummary(double simulation_start) const {
    if (param_.verbosity == 0) return;

    IO::messagef("simu time is %g seconds.\n",
                 GetTime() - simulation_start);
    IO::messagef("gate-batch runner: %u gate batches, %u restore passes, "
                 "%u raw gates, %u executable gates, "
                 "%u qubit swaps.\n",
                 simulation_stats_.num_gate_batches,
                 simulation_stats_.num_restore_passes,
                 unsigned(pending_gates_.size()),
                 simulation_stats_.num_executable_gates,
                 simulation_stats_.num_swaps);
    if (param_.verbosity > 1) {
      IO::messagef("breakdown: plan %g s, swap passes %g s, fuse %g s, "
                   "gate passes %g s.\n",
                   simulation_stats_.plan_seconds,
                   simulation_stats_.swap_seconds,
                   simulation_stats_.fuse_seconds,
                   simulation_stats_.gate_seconds);
    }
  }

  // State owned or borrowed by one invocation of the public static Run.
  // The instance never escapes Run, so the borrowed parameter and state
  // storage remain valid for its complete lifetime.
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
