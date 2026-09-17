#ifndef QSIM_QUBIT_REMAP_H_
#define QSIM_QUBIT_REMAP_H_

#include <cstdint>
#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "qubit_layout.h"

namespace qsim {

struct QubitSwap { unsigned physical_a; unsigned physical_b; };

class QubitRemapper {
 public:
  explicit QubitRemapper(unsigned n = 0) : layout_(n) {}
  QubitLayout& layout() { return layout_; }
  const QubitLayout& layout() const { return layout_; }
  std::vector<QubitSwap> MakeLocal(const std::vector<unsigned>& logical,
                                   unsigned local_slots) {
    std::vector<QubitSwap> swaps;
    if (logical.size() > local_slots) return swaps;

    // Keep every required qubit already in the local region at its current
    // physical position.  The previous implementation packed every batch
    // into slots [0, batch_size), needlessly evicting resident qubits and
    // turning otherwise cheap batches into full-state swap passes.
    std::vector<bool> reserved(local_slots, false);
    std::vector<std::pair<unsigned, unsigned>> assignments;
    assignments.reserve(logical.size());
    std::set<unsigned> assigned;
    for (auto q : logical) {
      const unsigned physical = layout_.LogicalToPhysical(q);
      if (physical < local_slots) {
        reserved[physical] = true;
        assignments.push_back({q, physical});
        assigned.insert(q);
      }
    }

    // Assign new qubits to the remaining local slots.  A deterministic slot
    // order keeps the layout stable across runs and makes scheduling results
    // reproducible.
    for (auto q : logical) {
      if (assigned.count(q) != 0) continue;
      auto slot = std::find(reserved.begin(), reserved.end(), false);
      if (slot == reserved.end()) return {};
      const unsigned physical_slot = static_cast<unsigned>(
          slot - reserved.begin());
      *slot = true;
      assignments.push_back({q, physical_slot});
      assigned.insert(q);
    }

    for (const auto& assignment : assignments) {
      const unsigned wanted = layout_.LogicalToPhysical(assignment.first);
      const unsigned target = assignment.second;
      if (wanted == target) continue;
      swaps.push_back({wanted, target});
      layout_.SwapPhysical(wanted, target);
    }
    return swaps;
  }
 private:
  QubitLayout layout_;
};

// Permutes a normal-order interleaved complex<float/double> state. SIMD
// backends keep their own internal order, so this primitive is intended for
// portable tile tools and for callers that explicitly normalize the state.
template <typename FP>
inline void ApplyBitSwap(FP* state, unsigned qubits, unsigned a, unsigned b) {
  if (a == b) return;
  const uint64_t n = uint64_t{1} << qubits;
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t j = i ^ (((i >> a) ^ (i >> b)) & 1) * ((uint64_t{1} << a) | (uint64_t{1} << b));
    if (i < j) { std::swap(state[2 * i], state[2 * j]); std::swap(state[2 * i + 1], state[2 * j + 1]); }
  }
}
}
#endif
