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

#ifndef UNITARYSPACE_H_
#define UNITARYSPACE_H_

#ifdef _WIN32
  #include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "util.h"

namespace qsim {

namespace detail {

inline void do_not_free(void*) noexcept {}

}  // namespace detail

namespace unitary {

/**
 * Abstract class containing routines for general unitary matrix manipulations.
 * A "Basic" implementation is provided (no AVX or SSE).
 */
template <typename Impl, typename For, typename FP>
class UnitarySpace {
 public:
  using fp_type = FP;
  using Unitary = std::unique_ptr<fp_type, decltype(&free)>;

  template <typename... ForArgs>
  UnitarySpace(uint64_t raw_size, unsigned num_qubits, ForArgs&&... args)
      : for_(args...), raw_size_(raw_size), num_qubits_(num_qubits) {}

  Unitary CreateUnitary() const {
    auto mat_size = sizeof(fp_type) * raw_size_;
    #ifdef _WIN32
      return Unitary((fp_type*) _aligned_malloc(mat_size, 64), &_aligned_free);
    #else
      void* p = nullptr;
      if (posix_memalign(&p, 64, mat_size) == 0) {
        return Unitary((fp_type*) p, &free);
      } else {
        return Unitary(nullptr, &free);
      }
    #endif
  }

  static Unitary CreateUnitary(fp_type* p) {
    return Unitary(p, &detail::do_not_free);
  }

  static Unitary NullState() {
    return Unitary(nullptr, &free);
  }

  uint64_t Size() const {
    return uint64_t{1} << num_qubits_;
  }

  uint64_t RawSize() const {
    return raw_size_;
  }

  static fp_type* RawData(Unitary& state) {
    return state.get();
  }

  static const fp_type* RawData(const Unitary& state) {
    return state.get();
  }

  static bool IsNull(const Unitary& state) {
    return state.get() == nullptr;
  }

  void CopyUnitary(const Unitary& src, Unitary& dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const Unitary& src, Unitary& dest) {
      dest.get()[i] = src.get()[i];
    };

    for_.Run(raw_size_, f, src, dest);
  }

  For for_;
  uint64_t raw_size_;
  unsigned num_qubits_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_H_
