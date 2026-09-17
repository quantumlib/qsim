#ifndef QSIM_QUBIT_LAYOUT_H_
#define QSIM_QUBIT_LAYOUT_H_

#include <algorithm>
#include <numeric>
#include <vector>

namespace qsim {

class QubitLayout {
 public:
  explicit QubitLayout(unsigned n = 0) { Reset(n); }
  void Reset(unsigned n) { logical_to_physical_.resize(n); physical_to_logical_.resize(n); std::iota(logical_to_physical_.begin(), logical_to_physical_.end(), 0); std::iota(physical_to_logical_.begin(), physical_to_logical_.end(), 0); }
  unsigned size() const { return logical_to_physical_.size(); }
  unsigned LogicalToPhysical(unsigned q) const { return logical_to_physical_.at(q); }
  unsigned PhysicalToLogical(unsigned q) const { return physical_to_logical_.at(q); }
  const std::vector<unsigned>& logical_to_physical() const { return logical_to_physical_; }
  const std::vector<unsigned>& physical_to_logical() const { return physical_to_logical_; }
  void SwapPhysical(unsigned a, unsigned b) {
    if (a == b) return;
    unsigned la = physical_to_logical_.at(a), lb = physical_to_logical_.at(b);
    std::swap(physical_to_logical_[a], physical_to_logical_[b]);
    logical_to_physical_[la] = b; logical_to_physical_[lb] = a;
  }
 private:
  std::vector<unsigned> logical_to_physical_, physical_to_logical_;
};
}
#endif
