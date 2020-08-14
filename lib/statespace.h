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

#ifdef _WIN32
  #include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "util.h"

namespace qsim {

namespace detail {

inline void do_not_free(void*) noexcept {}

}  // namespace detail

// Routines for state-vector manipulations.
template <typename Impl, typename For, typename FP>
class StateSpace {
 public:
  using fp_type = FP;
  using State = std::unique_ptr<fp_type, decltype(&free)>;

  struct MeasurementResult {
    uint64_t mask;
    uint64_t bits;
    std::vector<unsigned> bitstring;
    bool valid;
  };

  template <typename... ForArgs>
  StateSpace(uint64_t raw_size, unsigned num_qubits, ForArgs&&... args)
      : for_(args...), raw_size_(raw_size), num_qubits_(num_qubits) {}

  State CreateState() const {
    auto vector_size = sizeof(fp_type) * raw_size_;
    #ifdef _WIN32
      return State((fp_type*) _aligned_malloc(vector_size, 64), &_aligned_free);
    #else
      void* p = nullptr;
      if (posix_memalign(&p, 64, vector_size) == 0) {
        return State((fp_type*) p, &free);
      } else {
        return State(nullptr, &free);
      }
    #endif
  }

  static State CreateState(fp_type* p) {
    return State(p, &detail::do_not_free);
  }

  static State NullState() {
    return State(nullptr, &free);
  }

  uint64_t Size() const {
    return uint64_t{1} << num_qubits_;
  }

  uint64_t RawSize() const {
    return raw_size_;
  }

  static fp_type* RawData(State& state) {
    return state.get();
  }

  static const fp_type* RawData(const State& state) {
    return state.get();
  }

  static bool IsNull(const State& state) {
    return state.get() == nullptr;
  }

  static void Swap(State& state1, State& state2) {
    std::swap(state1, state2);
  }

  void CopyState(const State& src, State& dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& src, State& dest) {
      dest.get()[i] = src.get()[i];
    };

    for_.Run(raw_size_, f, src, dest);
  }

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

  For for_;
  uint64_t raw_size_;
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // STATESPACE_H_
