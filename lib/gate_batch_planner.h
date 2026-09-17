#ifndef QSIM_GATE_BATCH_PLANNER_H_
#define QSIM_GATE_BATCH_PLANNER_H_

#include <algorithm>
#include <cstdint>
#include <vector>
#include <set>

#include "operation_base.h"
#include "operation.h"

namespace qsim {
template <typename Operation>
class GateBatchPlanner {
 public:
  struct Batch {
    std::size_t first = 0, last = 0;
    unsigned qubits = 0;
    bool measurement_boundary = false;
    std::vector<unsigned> logical_qubits;
    // Non-contiguous batches are causally executable disjoint gates.  The
    // legacy first/last range remains populated for contiguous batches.
    std::vector<std::size_t> indices;
  };
  GateBatchPlanner(const std::vector<Operation>& ops, unsigned local_qubits, unsigned max_fused)
      : ops_(ops), local_qubits_(local_qubits), max_fused_(std::max(1U, max_fused)) {}
  std::vector<Batch> Plan() const {
    for (const auto& op : ops_) {
      if (OpGetAlternative<Measurement>(op)) return PlanContiguous();
    }
    return PlanCostAware();
  }

 private:
  struct GateInfo {
    std::vector<unsigned> qubits;
  };

  GateInfo GetGateInfo(const Operation& op) const {
    using FP = OpFpType<Operation>;
    GateInfo info;
    if (const auto* gate = OpGetAlternative<Gate<FP>>(op)) {
      info.qubits = gate->qubits;
    } else if (const auto* controlled =
                   OpGetAlternative<ControlledGate<FP>>(op)) {
      info.qubits = controlled->qubits;
      info.qubits.insert(info.qubits.end(), controlled->controlled_by.begin(),
                         controlled->controlled_by.end());
    }
    std::sort(info.qubits.begin(), info.qubits.end());
    info.qubits.erase(std::unique(info.qubits.begin(), info.qubits.end()),
                      info.qubits.end());
    return info;
  }

  std::vector<Batch> PlanContiguous() const {
    std::vector<Batch> out;
    Batch current;
    for (std::size_t i = 0; i < ops_.size(); ++i) {
      const auto* measurement_op = OpGetAlternative<Measurement>(ops_[i]);
      std::vector<unsigned> op_qubits = GetGateInfo(ops_[i]).qubits;
      if (measurement_op) op_qubits = measurement_op->qubits;
      std::set<unsigned> candidate(current.logical_qubits.begin(), current.logical_qubits.end());
      candidate.insert(op_qubits.begin(), op_qubits.end());
      unsigned q = candidate.size();
      bool measurement = measurement_op != nullptr;
      // max_fused_size belongs to the fuser, not to batch length. A batch may
      // contain many causal gates as long as its live qubit region remains
      // local; splitting it by operation count causes needless remaps.
      if (current.first != current.last && (measurement || q > local_qubits_)) {
        out.push_back(current); current = Batch{};
        candidate.clear();
        candidate.insert(op_qubits.begin(), op_qubits.end());
        q = candidate.size();
      }
      if (current.first == current.last) current.first = i;
      current.last = i + 1; current.qubits = q;
      current.logical_qubits.assign(candidate.begin(), candidate.end());
      current.indices.push_back(i);
      current.measurement_boundary |= measurement;
      if (measurement) { out.push_back(current); current = Batch{}; }
    }
    if (current.first != current.last) out.push_back(current);
    return out;
  }

  static bool Contains(const std::vector<unsigned>& set, unsigned q) {
    return std::binary_search(set.begin(), set.end(), q);
  }

  Batch EvaluateCandidate(const std::vector<GateInfo>& info,
                          const std::vector<bool>& pending,
                          const std::vector<unsigned>& candidate) const {
    Batch batch;
    std::vector<bool> blocked;
    unsigned max_qubit = 0;
    for (const auto& gate : info)
      for (auto q : gate.qubits) max_qubit = std::max(max_qubit, q + 1);
    blocked.assign(max_qubit, false);
    std::set<unsigned> used;
    for (std::size_t i = 0; i < info.size(); ++i) {
      if (!pending[i]) continue;
      bool fits = true;
      for (auto q : info[i].qubits) {
        if (!Contains(candidate, q) || blocked[q]) { fits = false; break; }
      }
      if (fits) {
        batch.indices.push_back(i);
        used.insert(info[i].qubits.begin(), info[i].qubits.end());
      } else {
        for (auto q : info[i].qubits) blocked[q] = true;
      }
    }
    batch.logical_qubits.assign(used.begin(), used.end());
    batch.qubits = batch.logical_qubits.size();
    if (!batch.indices.empty()) {
      batch.first = *std::min_element(batch.indices.begin(), batch.indices.end());
      batch.last = *std::max_element(batch.indices.begin(), batch.indices.end()) + 1;
    }
    return batch;
  }

  std::vector<Batch> PlanCostAware() const {
    std::vector<GateInfo> info;
    info.reserve(ops_.size());
    for (const auto& op : ops_) info.push_back(GetGateInfo(op));
    std::vector<bool> pending(ops_.size(), true);
    std::vector<Batch> result;
    while (std::find(pending.begin(), pending.end(), true) != pending.end()) {
      std::set<unsigned> all_resident;
      // A resident candidate is approximated by the qubits used by the last
      // selected batch. This preserves locality without requiring the planner
      // to own the execution state's complete physical layout.
      if (!result.empty())
        all_resident.insert(result.back().logical_qubits.begin(),
                            result.back().logical_qubits.end());
      std::vector<std::vector<unsigned>> candidates;
      candidates.emplace_back();
      if (!all_resident.empty())
        candidates.emplace_back(all_resident.begin(), all_resident.end());
      unsigned seeds = 0;
      for (std::size_t i = 0; i < info.size() && seeds < 16; ++i) {
        if (!pending[i] || info[i].qubits.empty()) continue;
        candidates.push_back(info[i].qubits);
        ++seeds;
      }
      Batch best;
      double best_score = -1e30;
      for (auto candidate : candidates) {
        std::sort(candidate.begin(), candidate.end());
        candidate.erase(std::unique(candidate.begin(), candidate.end()),
                        candidate.end());
        if (candidate.size() > local_qubits_) continue;
        // Grow the candidate greedily. EvaluateCandidate applies the causal
        // blocking rule after the candidate has been formed.
        for (std::size_t i = 0; i < info.size(); ++i) {
          if (!pending[i]) continue;
          std::set<unsigned> expanded(candidate.begin(), candidate.end());
          expanded.insert(info[i].qubits.begin(), info[i].qubits.end());
          if (expanded.size() <= local_qubits_) {
            candidate.assign(expanded.begin(), expanded.end());
          }
        }
        auto batch = EvaluateCandidate(info, pending, candidate);
        if (batch.indices.empty()) continue;
        const double swap_estimate =
            batch.logical_qubits.size() > all_resident.size()
                ? batch.logical_qubits.size() - all_resident.size() : 0;
        const double score = static_cast<double>(batch.indices.size()) -
                             0.35 * swap_estimate;
        if (score > best_score) { best_score = score; best = std::move(batch); }
      }
      if (best.indices.empty()) {
        auto first = std::find(pending.begin(), pending.end(), true) - pending.begin();
        best.indices = {static_cast<std::size_t>(first)};
        best.logical_qubits = info[first].qubits;
        best.qubits = best.logical_qubits.size();
        best.first = first; best.last = first + 1;
      }
      for (auto i : best.indices) pending[i] = false;
      result.push_back(std::move(best));
    }
    return result;
  }

  const std::vector<Operation>& ops_; unsigned local_qubits_, max_fused_;
};
}
#endif
