// Copyright 2019 Google LLC. All Rights Reserved.
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

#ifndef STATESPACE_H_
#define STATESPACE_H_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "util.h"
#include "vectorspace.h"

namespace qsim {

// Routines for state-vector manipulations.
template <typename Impl, typename For, typename FP>
class StateSpace : public VectorSpace<For, FP> {
 private:
  using Base = VectorSpace<For, FP>;

 public:
  using fp_type = typename Base::fp_type;
  using State = typename Base::Vector;

  struct MeasurementResult {
    uint64_t mask;
    uint64_t bits;
    std::vector<unsigned> bitstring;
    bool valid;
  };

  template <typename... ForArgs>
  StateSpace(uint64_t raw_size, unsigned num_qubits, ForArgs&&... args)
      : Base(raw_size, args...), num_qubits_(num_qubits) {}

  State CreateState() const {
    return Base::CreateVector();
  }

  static State CreateState(fp_type* p, uint64_t size) {
    return Base::CreateVector(p, size);
  }

  static State NullState() {
    return Base::NullVector();
  }

  bool CopyState(const State& src, State& dest) const {
    return Base::CopyVector(src, dest);
  }

  uint64_t Size() const {
    return uint64_t{1} << num_qubits_;
  }

  double Norm(const State& state) const {
    auto partial_norms = static_cast<const Impl&>(*this).PartialNorms(state);

    double norm = !partial_norms.empty() ? partial_norms[0] : std::nan("");
    for (std::size_t i = 1; i < partial_norms.size(); ++i) {
      norm += partial_norms[i];
    }

    return norm;
  }

  template <typename RGen>
  MeasurementResult Measure(const std::vector<unsigned>& qubits,
                            RGen& rgen, State& state) const {
    auto result =
        static_cast<const Impl&>(*this).VirtualMeasure(qubits, rgen, state);

    if (result.valid) {
      static_cast<const Impl&>(*this).CollapseState(result, state);
    }

    return result;
  }

  template <typename RGen>
  MeasurementResult VirtualMeasure(const std::vector<unsigned>& qubits,
                                   RGen& rgen, const State& state) const {
    MeasurementResult result;

    result.valid = true;
    result.mask = 0;

    if (state.size() != Base::raw_size_) {
      result.valid = false;
      return result;
    }

    for (auto q : qubits) {
      if (q >= num_qubits_) {
        result.valid = false;
        return result;
      }

      result.mask |= uint64_t{1} << q;
    }

    auto partial_norms = static_cast<const Impl&>(*this).PartialNorms(state);

    for (std::size_t i = 1; i < partial_norms.size(); ++i) {
      partial_norms[i] += partial_norms[i - 1];
    }

    auto norm = partial_norms.back();
    auto r = RandomValue(rgen, norm);

    unsigned m = 0;
    while (r > partial_norms[m]) ++m;
    if (m > 0) {
      r -= partial_norms[m - 1];
    }

    result.bits = static_cast<const Impl&>(*this).FindMeasuredBits(
        m, r, result.mask, state);

    result.bitstring.reserve(qubits.size());
    result.bitstring.resize(0);

    for (auto q : qubits) {
      result.bitstring.push_back((result.bits >> q) & 1);
    }

    return result;
  }

 protected:
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // STATESPACE_H_
