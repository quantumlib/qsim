// Gate-batched cache-local simulation runner. Follows the original
// proposal as closely as possible (see original_proposal.md).
//
// Data hierarchy:
//   Circuit: the full program, as an ordered list of raw gates.
//   State block: a cache-sized slice of 2^block_qubits amplitudes; one
//                gate batch runs against every state block.
//   SIMD chunk: the vectorized unit inside a state block - 2^chunk_qubits
//               complex amplitudes, architecture-dependent width (NEON/SSE:
//               4, AVX2: 8, AVX512: 16). Remapping moves whole chunks.
//
// A gate batch ties this together: the planner selects a compatible set
// of circuit gates, remaps their logical qubits into physical block
// positions, fuses them, and runs the result on every state block.
//
// Algorithm:
//   Move the most-used logical qubits into the low block up front
//   (PlaceHotQubitsInFixedZone) - free, since the all-zero initial state
//   is permutation-invariant.
//
//   while gates remain:
//     Select the next gate batch (GateBatchPlanner::PlanNextGateBatch):
//       score a bounded set of candidate qubit sets - the greedy
//       first-fit scan, the qubits already resident, and a few sets
//       seeded from upcoming gates - and keep the best.
//     Remap the batch's qubits into physical positions [0, block_qubits)
//       with one pass of disjoint transpositions (BuildBatchSwapPairs,
//       ApplySwapsToState). The low chunk_qubits positions are pinned and
//       never swapped; every other qubit draws from a bounded remap budget
//       (GateBatchPlanner::RemapSlotCapacity).
//     Fuse the batch's gates on physical qubits with the standard fuser
//       (FuseBatchGates); max_fused_size == 0 disables fusion.
//     Execute every fused gate on every state block, blocks in parallel,
//       each with its own sequential simulator (ExecuteGateBatchOnBlocks).
//
//   Restore identity qubit order with a final sequence of disjoint-
//   transposition passes (RestoreIdentityQubitOrder).
//
// Gates are scheduled raw: there is no global pre-fusion, 1q-chain
// peephole, or diagonal-gate phase kernel/bundle. All fusion happens
// inside the per-batch fuser above.

#ifndef RUN_QSIM_GATE_BATCH_H_
#define RUN_QSIM_GATE_BATCH_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "gate.h"
#include "qubit_mapped_state.h"
#include "matrix.h"
#include "operation_base.h"
#include "qubit_remap.h"
#include "util.h"

namespace qsim {

namespace gate_batch_internal {

// Internal scheduling representation of one raw circuit gate.
template <typename FP>
struct PendingGate {
  PendingGate(std::vector<unsigned> qubits, Matrix<FP> gate_matrix)
      : logical_qubits(std::move(qubits)),
        matrix(std::move(gate_matrix)) {}

  bool IsPending() const { return !applied_; }
  void MarkApplied() { applied_ = true; }
  std::size_t Arity() const { return logical_qubits.size(); }

  std::vector<unsigned> logical_qubits;  // ascending
  Matrix<FP> matrix;

 private:
  bool applied_ = false;
};

// A fused (or passthrough) gate ready to execute inside a state block.
template <typename FP>
struct ExecutableGate {
  std::vector<unsigned> physical_qubits;  // ascending
  Matrix<FP> matrix;
};

// Geometry of the cache-blocked state: an n-qubit state is viewed as
// 2^(n - L) contiguous blocks of 2^L amplitudes.
template <typename StateSpace>
struct BlockPartition {
  BlockPartition(unsigned state_qubits, unsigned requested_block_qubits)
      : num_state_qubits(state_qubits),
        block_qubits(std::min(requested_block_qubits, state_qubits)),
        floats_per_block(StateSpace::MinSize(block_qubits)),
        num_blocks(int64_t{1} << (state_qubits - block_qubits)) {}

  unsigned num_state_qubits;    // n
  unsigned block_qubits;        // L
  uint64_t floats_per_block;    // state floats per block (SIMD layout)
  int64_t num_blocks;           // 2^(n - L)
};

// Counters and phase timings accumulated across gate batches. Gate batches
// and identity-restoration passes are counted separately because only gate
// batches make circuit progress; restore passes are pure overhead. Timers
// are filled only at verbosity > 1.
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

// Buffers reused by every gate batch.
template <typename FP>
struct GateBatchWorkspace {
  std::vector<QubitSwap> swap_pairs;
  std::vector<ExecutableGate<FP>> executable_gates;
};

// Logical qubits selected for one gate batch, with eviction-aware remap-slot
// accounting. Logical qubits resident below the eviction floor join for free.
struct GateBatchQubitSet {
  bool Contains(unsigned logical_qubit) const {
    return contains_qubit[logical_qubit];
  }

  std::vector<char> contains_qubit;
  unsigned remap_slots_used = 0;
  unsigned remap_slot_capacity = 0;
};

// The outcome of planning one gate batch, built without mutating gates,
// layout, or state.
struct GateBatchPlan {
  bool HasGates() const { return !gate_indices.empty(); }
  std::size_t NumGates() const { return gate_indices.size(); }
  bool UsesQubit(unsigned logical_qubit) const {
    return uses_qubit[logical_qubit];
  }
  bool operator<(const GateBatchPlan& other) const {
    return score < other.score;
  }

  std::vector<std::size_t> gate_indices;
  std::vector<char> uses_qubit;
  unsigned required_swaps = 0;
  double score = 0.0;
};

}  // namespace gate_batch_internal

template <typename IO, typename Fuser, typename Factory,
          typename SeqSimulator>
class QSimGateBatchRunner final {
 public:
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using QubitMappedState = qsim::QubitMappedState<State>;
  using fp_type = typename StateSpace::fp_type;
  using SeqStateSpace = typename SeqSimulator::StateSpace;

  static_assert(std::is_same_v<fp_type, float>,
                "QSimGateBatchRunner requires a float state space.");
  static_assert(std::is_same_v<fp_type, typename SeqStateSpace::fp_type>,
                "State spaces must use the same floating-point type.");
  static_assert(StateSpace::kChunkQubits == SeqStateSpace::kChunkQubits,
                "State spaces must use the same SIMD chunk layout.");

  struct Parameter : public Fuser::Parameter {
    Parameter() { this->max_fused_size = 3; }

    unsigned block_qubits = 19;
  };

  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, State& state) {
    // Keep the conventional qsim static entry point while putting all
    // mutable execution state in a private, one-shot runner instance.
    (void) factory;  // Present for API compatibility with other runners.
    QubitLayout layout(circuit.num_qubits);
    QSimGateBatchRunner runner(param, circuit.num_qubits, state, layout);
    return runner.SimulateCircuit(circuit, true);
  }

  // Runs directly on a mapped state and leaves its physical storage in the
  // returned logical-to-physical layout, avoiding final restoration.
  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, QubitMappedState& state) {
    (void) factory;
    QSimGateBatchRunner runner(param, circuit.num_qubits, state.state,
                              state.layout);
    return runner.SimulateCircuit(circuit, false);
  }

 private:
  using PendingGate = gate_batch_internal::PendingGate<fp_type>;
  using ExecutableGate = gate_batch_internal::ExecutableGate<fp_type>;
  using BlockPartition = gate_batch_internal::BlockPartition<StateSpace>;
  using SimulationStats = gate_batch_internal::SimulationStats;
  using GateBatchWorkspace =
      gate_batch_internal::GateBatchWorkspace<fp_type>;
  using GateBatchQubitSet = gate_batch_internal::GateBatchQubitSet;
  using GateBatchPlan = gate_batch_internal::GateBatchPlan;

  QSimGateBatchRunner(const Parameter& param, unsigned num_qubits,
                      State& state, QubitLayout& layout)
      : param_(param),
        partition_(num_qubits, param.block_qubits),
        state_data_(state.get()),
        chunk_qubits_(StateSpace::kChunkQubits),
        seq_sim_(1),
        layout_(layout),
        gate_batch_planner_(partition_.num_state_qubits,
                            partition_.block_qubits, chunk_qubits_) {
    assert(partition_.block_qubits >= chunk_qubits_);
    assert(layout_.NumQubits() == partition_.num_state_qubits);
  }

  template <typename Circuit>
  bool SimulateCircuit(const Circuit& circuit, bool restore_qubit_order) {
    using Op = typename std::decay_t<decltype(circuit.ops)>::value_type;

    const auto prepare_start = StartPhaseTimer();
    if (!PreparePendingGates(circuit)) return false;
    LogPreparationTime(prepare_start);

    const auto simulation_start = param_.verbosity > 0 ? GetTime() : 0.0;
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
    LogSimulationSummary(simulation_start);
    return true;
  }

  // ======== Phase timing ========

  double StartPhaseTimer() const {
    return param_.verbosity > 1 ? GetTime() : 0.0;
  }

  void AccumulatePhaseSeconds(double start, double& seconds) const {
    if (param_.verbosity > 1) seconds += GetTime() - start;
  }

  void LogPreparationTime(double prepare_start) const {
    if (param_.verbosity <= 1) return;
    IO::messagef("prepare time is %g seconds.\n",
                 GetTime() - prepare_start);
  }

  // ======== Shared gate utilities ========

  // Sorts a gate's qubits ascending, permuting `matrix` to match.
  static void NormalizeGateQubitOrder(std::vector<unsigned>& qubits,
                                      Matrix<fp_type>& matrix) {
    if (qubits.size() < 2) return;
    auto permutation = NormalToGateOrderPermutation(qubits);
    if (!permutation.empty()) {
      MatrixShuffle(permutation, unsigned(qubits.size()), matrix);
      std::sort(qubits.begin(), qubits.end());
    }
  }

  // ======== Pending-gate preparation ========

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
      pending_gates_.emplace_back(std::move(logical_qubits),
                                  std::move(matrix));
    }
    return true;
  }

  // ======== Gate-batch planning ========

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
  class GateBatchPlanner {
   public:
    GateBatchPlanner(unsigned num_state_qubits, unsigned block_qubits,
                     unsigned chunk_qubits)
        : num_logical_qubits_(num_state_qubits),
          block_qubits_(block_qubits),
          chunk_qubits_(chunk_qubits),
          is_qubit_blocked_(num_state_qubits) {}

    GateBatchPlan PlanNextGateBatch(const std::vector<PendingGate>& gates,
                                    const QubitLayout& layout) {
      std::vector<GateBatchPlan> candidate_plans;
      candidate_plans.reserve(2 + kMaxGateSeeds);

      // First-fit greedy candidate.
      candidate_plans.push_back(PlanGateBatchFromQubitSet(
          gates, layout, MakeEmptyQubitSet()));

      // Zero-swap resident candidate.
      candidate_plans.push_back(PlanGateBatchFromQubitSet(
          gates, layout, MakeResidentQubitSet(layout)));

      for (std::size_t seed_idx : CollectSeedGateIndices(gates)) {
        auto qubit_set = MakeEmptyQubitSet();
        if (!TryAdmitQubits(gates[seed_idx].logical_qubits,
                            layout, qubit_set)) {
          continue;
        }
        // Candidate seeded by a later pending gate.
        candidate_plans.push_back(PlanGateBatchFromQubitSet(
            gates, layout, std::move(qubit_set)));
      }

      // max_element returns the first maximum, so ties preserve candidate
      // priority: greedy, resident, then seeded plans in circuit order.
      return std::move(*std::max_element(
          candidate_plans.begin(), candidate_plans.end()));
    }

    unsigned EvictionFloor() const { return ComputeEvictionFloor(); }

   private:
    // How many pending gates beyond the first (which already leads the
    // greedy baseline) get to seed their own candidate qubit set.
    static constexpr unsigned kMaxGateSeeds = 4;

    // Keep enough remap positions for an ordinary two-qubit gate even when
    // the block is too small to reach kMinEvictionFloor.
    static constexpr unsigned kMinRemapSlots = 2;

    // Lowest eviction position allowed in a multi-block gate batch: swap-pass
    // spans are 2^(floor - chunk_qubits) chunks, so floor 11 keeps every
    // span at least 16 KiB for float states and at streaming bandwidth.
    // See RemapSlotCapacity.
    static constexpr unsigned kMinEvictionFloor = 11;

    // Starting any remapping incurs a full-state pass. Additional pairs share
    // that pass and therefore carry a smaller marginal cost. With these
    // weights, eight pairs offset one additional gate.
    static constexpr double kSwapPassCost = 0.5;
    static constexpr double kSwapPairCost = 1.0 / 16.0;

    GateBatchQubitSet MakeEmptyQubitSet() const {
      GateBatchQubitSet qubit_set;
      qubit_set.contains_qubit.assign(num_logical_qubits_, 0);
      qubit_set.remap_slot_capacity = RemapSlotCapacity();
      return qubit_set;
    }

    // Positions below the eviction floor are fixed residents and join a set
    // for free. Positions at or above it consume one remap slot: a low
    // resident protects a potential victim, while a high qubit requires a
    // victim. Keeping the floor at kMinEvictionFloor makes every remap span
    // at least 16 KiB for the default L=19. Small blocks retain at least two
    // remap slots so an ordinary two-qubit gate can make progress.
    unsigned RemapSlotCapacity() const {
      return block_qubits_ - ComputeEvictionFloor();
    }

    unsigned ComputeEvictionFloor() const {
      if (block_qubits_ == num_logical_qubits_) return block_qubits_;

      const auto full = block_qubits_ > chunk_qubits_
                            ? block_qubits_ - chunk_qubits_
                            : 0u;
      const auto depth_capped = block_qubits_ > kMinEvictionFloor
                                    ? block_qubits_ - kMinEvictionFloor
                                    : 0u;
      const auto capacity =
          std::min(full, std::max(depth_capped, kMinRemapSlots));
      return block_qubits_ - capacity;
    }

    // The zero-swap candidate: exactly the qubits already resident in
    // the low block. Its occupancy may exceed the depth-capped capacity;
    // that is safe because none of these qubits needs a swap (no
    // evictions happen), and growth beyond them is still capacity-bound.
    GateBatchQubitSet MakeResidentQubitSet(
        const QubitLayout& layout) const {
      auto qubit_set = MakeEmptyQubitSet();
      for (unsigned q = 0; q < num_logical_qubits_; ++q) {
        if (layout.PhysicalPositionOf(q) < block_qubits_) {
          qubit_set.remap_slots_used +=
              AdditionalRemapSlotsFor(q, qubit_set, layout);
          qubit_set.contains_qubit[q] = 1;
        }
      }
      return qubit_set;
    }

    // Gate indices of the next kMaxGateSeeds pending gates after the
    // first pending one.
    std::vector<std::size_t> CollectSeedGateIndices(
        const std::vector<PendingGate>& gates) const {
      std::vector<std::size_t> seeds;
      bool skipped_first_pending = false;
      for (std::size_t idx = 0;
           idx < gates.size() && seeds.size() < kMaxGateSeeds; ++idx) {
        if (!gates[idx].IsPending()) continue;
        if (!skipped_first_pending) {
          skipped_first_pending = true;
          continue;
        }
        seeds.push_back(idx);
      }
      return seeds;
    }

    GateBatchPlan PlanGateBatchFromQubitSet(
        const std::vector<PendingGate>& gates, const QubitLayout& layout,
        GateBatchQubitSet qubit_set) {
      GrowQubitSetGreedily(gates, layout, qubit_set);
      return EvaluateQubitSet(gates, layout, qubit_set);
    }

    // First-fit growth: scan pending gates in circuit order, admitting
    // each gate's qubits when they fit and blocking them otherwise (the
    // same causal rule EvaluateQubitSet applies to the finished set).
    void GrowQubitSetGreedily(const std::vector<PendingGate>& gates,
                              const QubitLayout& layout,
                              GateBatchQubitSet& qubit_set) {
      ClearBlockedQubits();
      for (const PendingGate& gate : gates) {
        if (!gate.IsPending()) continue;
        if (AnyQubitBlocked(gate.logical_qubits) ||
            !TryAdmitQubits(gate.logical_qubits, layout, qubit_set)) {
          BlockQubits(gate.logical_qubits);
        }
      }
    }

    // Exact evaluation of a fixed qubit set: one causal-blocking scan over
    // the pending gates collects every gate the set can execute, then the
    // plan is scored.
    GateBatchPlan EvaluateQubitSet(
        const std::vector<PendingGate>& gates, const QubitLayout& layout,
        const GateBatchQubitSet& qubit_set) {
      GateBatchPlan plan;
      plan.uses_qubit.assign(num_logical_qubits_, 0);
      ClearBlockedQubits();

      for (std::size_t idx = 0; idx < gates.size(); ++idx) {
        const PendingGate& gate = gates[idx];
        if (!gate.IsPending()) continue;
        if (!AnyQubitBlocked(gate.logical_qubits) &&
            QubitSetContainsAll(gate.logical_qubits, qubit_set)) {
          plan.gate_indices.push_back(idx);
          for (unsigned q : gate.logical_qubits) {
            plan.uses_qubit[q] = 1;
          }
        } else {
          BlockQubits(gate.logical_qubits);
        }
      }

      plan.required_swaps = CountSwapsNeeded(plan, layout);
      const auto swap_cost =
          plan.required_swaps == 0
              ? 0.0
              : kSwapPassCost + kSwapPairCost * plan.required_swaps;
      plan.score = double(plan.NumGates()) - swap_cost;
      return plan;
    }

    // Remap-slot cost of admitting q. Fixed residents below the eviction
    // floor cost nothing. A resident in the eviction zone consumes a victim
    // position by protecting it, and a high qubit consumes one by entering.
    unsigned AdditionalRemapSlotsFor(
        unsigned q, const GateBatchQubitSet& qubit_set,
        const QubitLayout& layout) const {
      if (qubit_set.Contains(q)) return 0;
      return layout.PhysicalPositionOf(q) < ComputeEvictionFloor() ? 0 : 1;
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
          qubit_set.remap_slot_capacity) {
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
            layout.PhysicalPositionOf(q) >= block_qubits_) {
          ++count;
        }
      }
      return count;
    }

    void ClearBlockedQubits() {
      std::fill(is_qubit_blocked_.begin(), is_qubit_blocked_.end(), 0);
    }

    bool AnyQubitBlocked(const std::vector<unsigned>& qubits) const {
      return std::any_of(
          qubits.begin(), qubits.end(),
          [this](unsigned q) { return is_qubit_blocked_[q]; });
    }

    void BlockQubits(const std::vector<unsigned>& qubits) {
      for (unsigned q : qubits) is_qubit_blocked_[q] = 1;
    }

    unsigned num_logical_qubits_;
    unsigned block_qubits_;
    unsigned chunk_qubits_;
    std::vector<char> is_qubit_blocked_;
  };

  // ======== Gate-batch pipeline ========

  // Executes one complete gate batch following the proposal's loop body.
  // Returns the number of gates applied; 0 signals failure (fuser error, or
  // a schedule that cannot make progress). Planning is read-only; gates are
  // marked applied only after fusion succeeds, so a failed batch never
  // corrupts the schedule.
  template <typename Op>
  std::size_t ExecuteNextGateBatch(std::vector<Op>& batch_operations) {
    const auto block_qubits = partition_.block_qubits;

    // pick_maximum_number_of_gates_acting_on_low_qubits
    const auto plan_start = StartPhaseTimer();
    const auto plan = gate_batch_planner_.PlanNextGateBatch(
        pending_gates_, layout_);
    AccumulatePhaseSeconds(plan_start, simulation_stats_.plan_seconds);
    if (!plan.HasGates()) {
      IO::errorf("qsim_gate_batch: a gate does not fit the qubit set "
                 "of %u block qubits; use a larger block_qubits.\n",
                 block_qubits);
      return 0;
    }
    LogPlannedGateBatch(plan);

    // select_qubits_for_swap + swap_low_and_high_qubits
    BuildBatchSwapPairs(plan);
    LogGateBatchSwapSummary(plan);
    LogAppliedRemap();
    ApplySwapsToState(gate_batch_workspace_.swap_pairs);

    // fused_low_gates = fuse(low_gates)
    const auto fuse_start = StartPhaseTimer();
    BuildBatchOperations(plan, batch_operations);
    if (!FuseBatchGates(batch_operations)) {
      return 0;
    }
    AccumulatePhaseSeconds(fuse_start, simulation_stats_.fuse_seconds);
    simulation_stats_.num_executable_gates +=
        unsigned(gate_batch_workspace_.executable_gates.size());
    MarkGatesApplied(plan);

    // for i in 0..(2^num_high_qubits): apply all fused gates to block i
    const auto gates_start = StartPhaseTimer();
    ExecuteGateBatchOnBlocks(gate_batch_workspace_.executable_gates,
                             state_data_, partition_, seq_sim_);
    AccumulatePhaseSeconds(gates_start, simulation_stats_.gate_seconds);

    ++simulation_stats_.num_gate_batches;
    return plan.NumGates();
  }

  // ======== Gate-batch diagnostics ========

  // The last pair holds the lowest eviction position; a low floor means
  // short scattered spans in the swap pass (see qubit_remap.h).
  void LogGateBatchSwapSummary(const GateBatchPlan& plan) const {
    if (param_.verbosity <= 2 ||
        gate_batch_workspace_.swap_pairs.empty()) {
      return;
    }

    IO::messagef("gate batch %u: %u gates, %u swaps, evict floor %u\n",
                 simulation_stats_.num_gate_batches,
                 unsigned(plan.NumGates()),
                 unsigned(gate_batch_workspace_.swap_pairs.size()),
                 gate_batch_workspace_.swap_pairs.back().first);
  }

  // The planned gate batch before any remapping: every planned gate with its
  // logical qubits, then every distinct qubit the batch uses with its
  // current physical position; '*' marks qubits outside the block that
  // the remap is about to swap in.
  void LogPlannedGateBatch(const GateBatchPlan& plan) const {
    if (param_.verbosity <= 3) return;

    IO::messagef("gate batch %u plan: %u gates, %u swaps needed\n  gates:",
                 simulation_stats_.num_gate_batches,
                 unsigned(plan.NumGates()),
                 plan.required_swaps);
    for (std::size_t idx : plan.gate_indices) {
      const PendingGate& gate = pending_gates_[idx];
      IO::messagef(" #%u[", unsigned(idx));
      for (std::size_t i = 0; i < gate.Arity(); ++i) {
        IO::messagef(i == 0 ? "q%u" : ",q%u",
                     gate.logical_qubits[i]);
      }
      IO::messagef("]");
    }
    IO::messagef("\n  qubits:");
    for (unsigned q = 0; q < unsigned(plan.uses_qubit.size()); ++q) {
      if (!plan.UsesQubit(q)) continue;
      const auto position = layout_.PhysicalPositionOf(q);
      IO::messagef(" q%u@p%u%s", q, position,
                   position >= partition_.block_qubits ? "*" : "");
    }
    IO::messagef("\n");
  }

  // The remap just applied (the layout is already updated): each
  // transposition as "incoming qubit, its new<-old position, outgoing
  // qubit", then the low-block layout the batch's gates will use.
  void LogAppliedRemap() const {
    if (param_.verbosity <= 3) return;

    IO::messagef("  swaps:");
    if (gate_batch_workspace_.swap_pairs.empty()) IO::messagef(" none");
    for (const QubitSwap& pair : gate_batch_workspace_.swap_pairs) {
      IO::messagef(" [q%u in p%u<-p%u, q%u out]",
                   layout_.LogicalQubitAt(pair.first), pair.first,
                   pair.second, layout_.LogicalQubitAt(pair.second));
    }
    IO::messagef("\n  block:");
    for (unsigned p = 0; p < partition_.block_qubits; ++p) {
      IO::messagef(" q%u", layout_.LogicalQubitAt(p));
    }
    IO::messagef("\n");
  }

  // Extends the layout so every qubit the plan uses lands in physical
  // positions [0, L), recording the transpositions to apply. Eviction
  // walks down from L-1, skipping positions that hold used qubits; the
  // planner's slot accounting guarantees it never reaches the pinned
  // lane positions.
  void BuildBatchSwapPairs(const GateBatchPlan& plan) {
    gate_batch_workspace_.swap_pairs.clear();
    unsigned evict_position = partition_.block_qubits - 1;

    for (unsigned q = 0; q < unsigned(plan.uses_qubit.size()); ++q) {
      if (!plan.UsesQubit(q)) continue;
      const auto current_position = layout_.PhysicalPositionOf(q);
      if (current_position < partition_.block_qubits) continue;

      while (plan.UsesQubit(layout_.LogicalQubitAt(evict_position))) {
        --evict_position;
      }
      gate_batch_workspace_.swap_pairs.emplace_back(evict_position,
                                                     current_position);
      layout_.SwapPositions(evict_position, current_position);
      --evict_position;
    }
  }

  // Applies the transpositions to the state in one involution pass.
  void ApplySwapsToState(const std::vector<QubitSwap>& swap_pairs) {
    const auto swap_start = StartPhaseTimer();
    ApplyBitPairSwaps(state_data_, partition_.num_state_qubits, chunk_qubits_,
                      swap_pairs);
    AccumulatePhaseSeconds(swap_start, simulation_stats_.swap_seconds);
    simulation_stats_.num_swaps += unsigned(swap_pairs.size());
  }

  // Scores qubits by gate participation, weighted by gate arity.
  std::vector<uint64_t> ComputeQubitUsageScores() const {
    std::vector<uint64_t> scores(partition_.num_state_qubits, 0);
    for (const PendingGate& gate : pending_gates_) {
      const auto weight = gate.Arity();
      for (unsigned q : gate.logical_qubits) scores[q] += weight;
    }
    return scores;
  }

  // Returns the count highest-scoring qubits in deterministic rank order.
  static std::vector<unsigned> SelectHighestScoringQubits(
      const std::vector<uint64_t>& scores, unsigned first_qubit,
      unsigned count) {
    std::vector<unsigned> candidates;
    candidates.reserve(scores.size() - first_qubit);
    for (unsigned q = first_qubit; q < scores.size(); ++q) {
      candidates.push_back(q);
    }
    assert(count <= candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + count,
                      candidates.end(),
                      [&scores](unsigned a, unsigned b) {
                        return scores[a] != scores[b]
                                   ? scores[a] > scores[b] : a < b;
                      });
    candidates.resize(count);
    return candidates;
  }

  // Builds disjoint swaps that put the selected logical qubits in the fixed
  // physical zone and updates layout to match the state permutation.
  std::vector<QubitSwap>
  BuildFixedZonePlacementSwaps(const std::vector<unsigned>& fixed_qubits,
                               unsigned eviction_floor) {
    std::vector<char> is_fixed_qubit(layout_.NumQubits(), 0);
    for (unsigned q : fixed_qubits) is_fixed_qubit[q] = 1;

    std::vector<unsigned> victim_positions;
    std::vector<unsigned> incoming_qubits;
    victim_positions.reserve(fixed_qubits.size());
    incoming_qubits.reserve(fixed_qubits.size());

    for (unsigned p = chunk_qubits_; p < eviction_floor; ++p) {
      if (!is_fixed_qubit[layout_.LogicalQubitAt(p)]) {
        victim_positions.push_back(p);
      }
    }
    for (unsigned q : fixed_qubits) {
      if (layout_.PhysicalPositionOf(q) >= eviction_floor) {
        incoming_qubits.push_back(q);
      }
    }

    assert(victim_positions.size() == incoming_qubits.size());
    std::vector<QubitSwap> swap_pairs;
    swap_pairs.reserve(victim_positions.size());
    for (std::size_t i = 0; i < victim_positions.size(); ++i) {
      const auto victim_position = victim_positions[i];
      const auto incoming_position =
          layout_.PhysicalPositionOf(incoming_qubits[i]);
      swap_pairs.emplace_back(victim_position, incoming_position);
      layout_.SwapPositions(victim_position, incoming_position);
    }
    return swap_pairs;
  }

  void LogFixedZonePlacement(const std::vector<uint64_t>& usage_scores,
                             unsigned eviction_floor,
                             std::size_t num_initial_swaps) const {
    if (param_.verbosity <= 1) return;

    IO::messagef("fixed hot zone [%u,%u):", chunk_qubits_,
                 eviction_floor);
    for (unsigned p = chunk_qubits_; p < eviction_floor; ++p) {
      const auto q = layout_.LogicalQubitAt(p);
      IO::messagef(" q%u(%llu)", q,
                   static_cast<unsigned long long>(usage_scores[q]));
    }
    IO::messagef("; %u initial swaps.\n", unsigned(num_initial_swaps));
  }

  // Places the most frequently used logical qubits outside the in-chunk
  // physical zone [chunk_qubits, eviction_floor). In-chunk positions
  // remain untouched. Canonical runs restore qubit order at the end.
  void PlaceHotQubitsInFixedZone() {
    const auto eviction_floor = gate_batch_planner_.EvictionFloor();
    if (partition_.num_blocks == 1 ||
        eviction_floor <= chunk_qubits_) {
      return;
    }

    const auto num_fixed_qubits =
        eviction_floor - chunk_qubits_;
    const auto usage_scores = ComputeQubitUsageScores();
    const auto fixed_qubits =
        SelectHighestScoringQubits(usage_scores, chunk_qubits_,
                                   num_fixed_qubits);
    auto swap_pairs =
        BuildFixedZonePlacementSwaps(fixed_qubits, eviction_floor);
    LogFixedZonePlacement(usage_scores, eviction_floor, swap_pairs.size());
    ApplySwapsToState(swap_pairs);
  }

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
      const PendingGate& gate = pending_gates_[idx];

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
          0, time++, std::move(physical_qubits), {},
          std::move(physical_matrix), false};
      batch_operations.push_back(Op{std::move(physical_gate)});
    }
  }

  // Deferred until fusion has succeeded, so a failed batch leaves the
  // schedule untouched.
  void MarkGatesApplied(const GateBatchPlan& plan) {
    for (std::size_t idx : plan.gate_indices) {
      pending_gates_[idx].MarkApplied();
    }
  }

  // fuse(low_gates): runs the standard fuser over the batch's physical
  // gates (max_fused_size from param; the proposal suggests 2 or 3) and
  // copies the results into ExecutableGate records. max_fused_size == 0
  // is the no-fusion control mode.
  template <typename Op>
  bool FuseBatchGates(const std::vector<Op>& batch_operations) {
    auto& executable_gates =
        gate_batch_workspace_.executable_gates;
    executable_gates.clear();

    if (param_.max_fused_size == 0) {
      executable_gates.reserve(batch_operations.size());
      for (const auto& operation : batch_operations) {
        if (!AppendExecutableGate(operation)) return false;
      }
      return true;
    }

    auto fused_ops =
        Fuser::FuseGates(param_, partition_.block_qubits, batch_operations);
    if (fused_ops.size() == 0 && batch_operations.size() > 0) {
      IO::errorf("qsim_gate_batch: fuser failed on a gate batch.\n");
      return false;
    }

    executable_gates.reserve(fused_ops.size());
    for (const auto& operation : fused_ops) {
      if (!AppendExecutableGate(operation)) return false;
    }
    return true;
  }

  // Copies one (possibly fused) operation into an ExecutableGate,
  // normalizing qubit order. The copy must happen while the fuser's input
  // container is still alive (fused ops may point into it).
  template <typename Op>
  bool AppendExecutableGate(const Op& operation) {
    const std::vector<unsigned>* physical_qubits = nullptr;
    const Matrix<fp_type>* matrix = nullptr;

    if (const auto* fused_gate =
            OpGetAlternative<FusedGate<fp_type>>(operation)) {
      physical_qubits = &fused_gate->qubits;
      matrix = &fused_gate->matrix;
    } else if (const auto* raw_gate =
                   OpGetAlternative<Gate<fp_type>>(operation)) {
      physical_qubits = &raw_gate->qubits;
      matrix = &raw_gate->matrix;
    } else {
      IO::errorf("qsim_gate_batch: unsupported operation in gate batch.\n");
      return false;
    }

    ExecutableGate executable_gate{*physical_qubits, *matrix};
    NormalizeGateQubitOrder(executable_gate.physical_qubits,
                            executable_gate.matrix);
    gate_batch_workspace_.executable_gates.push_back(
        std::move(executable_gate));
    return true;
  }

  // The proposal's inner loops: for every block i, apply every fused
  // gate to the block while it is cache-resident. Blocks in parallel.
  // Unit-sized dynamic scheduling balances heterogeneous cores and gate
  // workloads consistently, including when the binary is run directly.
  static void ExecuteGateBatchOnBlocks(
      const std::vector<ExecutableGate>& executable_gates,
      fp_type* state_data,
      const BlockPartition& partition, SeqSimulator& seq_sim) {
    const int64_t num_blocks = partition.num_blocks;
    const auto floats_per_block = partition.floats_per_block;
    const auto block_qubits = partition.block_qubits;

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b = 0; b < num_blocks; ++b) {
      fp_type* block_data = state_data + uint64_t(b) * floats_per_block;
      auto block_view = SeqStateSpace::Create(block_data, block_qubits);

      for (const ExecutableGate& gate : executable_gates) {
        seq_sim.ApplyGate(gate.physical_qubits, gate.matrix.data(),
                          block_view);
      }
    }
  }

  // ======== Final cleanup ========

  void RestoreIdentityQubitOrder() {
    std::vector<QubitSwap> swap_pairs;

    while (layout_.BuildIdentityRestorationSwaps(swap_pairs)) {
      ApplySwapsToState(swap_pairs);
      ++simulation_stats_.num_restore_passes;
    }
  }

  // ======== Reporting ========

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
  const BlockPartition partition_;
  fp_type* const state_data_;
  const unsigned chunk_qubits_;  // Low amplitude bits inside a state chunk.
  SeqSimulator seq_sim_;
  std::vector<PendingGate> pending_gates_;
  QubitLayout& layout_;
  GateBatchPlanner gate_batch_planner_;
  GateBatchWorkspace gate_batch_workspace_;
  SimulationStats simulation_stats_;
};

}  // namespace qsim

#endif  // RUN_QSIM_GATE_BATCH_H_
