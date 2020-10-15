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
#include <vector>

#include "util.h"
#include "vectorspace.h"

namespace qsim {

/**
 * Abstract class containing context and routines for general state-vector
 * manipulations. "AVX", "Basic", and "SSE" implementations are provided.
 */
template <typename Impl, typename For, typename FP>
class StateSpace : public VectorSpace<Impl, For, FP> {
 private:
  using Base = VectorSpace<Impl, For, FP>;

 public:
  using fp_type = typename Base::fp_type;
  using State = typename Base::Vector;

  /**
   * The observed state from a Measurement gate.
   */
  struct MeasurementResult {
    /**
     * A bitmask of all qubits measured in this result. In this format, if the
     * qubit at index `i` is measured, the `i`th bit of `mask` is a one.
     */
    uint64_t mask;
    /**
     * A bitwise representation of the measured states. In this format, the
     * qubit at index `i` is represented by the `i`th bit of `bits`.
     * If `valid` is true, `mask` has already been applied to this field
     * (i.e. `bits == bits & mask`).
     */
    uint64_t bits;
    /**
     * Observed states of the measured qubits. This vector only includes qubits
     * specified by the associated Measurement gate.
     */
    std::vector<unsigned> bitstring;
    /**
     * Validation bit. If this is false, the measurement failed and all other
     * fields of the result are invalid.
     */
    bool valid;
  };

  template <typename... ForArgs>
  StateSpace(ForArgs&&... args) : Base(args...) {}

  double Norm(const State& state) const {
    auto partial_norms = static_cast<const Impl&>(*this).PartialNorms(state);

    double norm = partial_norms[0];
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
      static_cast<const Impl&>(*this).Collapse(result, state);
    }

    return result;
  }

  template <typename RGen>
  MeasurementResult VirtualMeasure(const std::vector<unsigned>& qubits,
                                   RGen& rgen, const State& state) const {
    MeasurementResult result;

    result.valid = true;
    result.mask = 0;

    for (auto q : qubits) {
      if (q >= state.num_qubits()) {
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
};

}  // namespace qsim

#endif  // STATESPACE_H_
