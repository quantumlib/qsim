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

#ifndef UNITARYSPACE_SSE_H_
#define UNITARYSPACE_SSE_H_

#include <smmintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

#include "unitaryspace.h"

namespace qsim {

namespace unitary {

/**
 * Object containing context and routines for unitary manipulations.
 * Unitary is a vectorized sequence of four real components followed by four
 * imaginary components. Four single-precison floating numbers can be loaded
 * into an SSE register.
 */
template <typename For>
struct UnitarySpaceSSE : public UnitarySpace<UnitarySpaceSSE<For>, For, float> {
 private:
  using Base = UnitarySpace<UnitarySpaceSSE<For>, For, float>;

 public:
  using Unitary = typename Base::Unitary;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit UnitarySpaceSSE(ForArgs&&... args) : Base(args...) {}

  static uint64_t MinRowSize(unsigned num_qubits) {
    return std::max(uint64_t{8}, 2 * (uint64_t{1} << num_qubits));
  };

  static uint64_t MinSize(unsigned num_qubits) {
    return Base::Size(num_qubits) * MinRowSize(num_qubits);
  };

  void SetAllZeros(Unitary& state) const {
    __m128 val0 = _mm_setzero_ps();

    auto f = [](unsigned n, unsigned m, uint64_t i, __m128 val0, fp_type* p) {
      _mm_store_ps(p + 8 * i, val0);
      _mm_store_ps(p + 8 * i + 4, val0);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f, val0, state.get());
  }

  void SetIdentity(Unitary& state) {
    SetAllZeros(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t row_size, fp_type* p) {
      p[row_size * i + (8 * (i / 4)) + (i % 4)] = 1;
    };

    uint64_t size = Base::Size(state.num_qubits());
    uint64_t row_size = MinRowSize(state.num_qubits());
    Base::for_.Run(size, f, row_size, state.get());
  }

  static std::complex<fp_type> GetEntry(const Unitary& state,
                                        uint64_t i, uint64_t j) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    uint64_t k = (8 * (j / 4)) + (j % 4);
    return std::complex<fp_type>(state.get()[row_size * i + k],
                                 state.get()[row_size * i + k + 4]);
  }

  static void SetEntry(Unitary& state, uint64_t i, uint64_t j,
                       const std::complex<fp_type>& ampl) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    uint64_t k = (8 * (j / 4)) + (j % 4);
    state.get()[row_size * i + k] = std::real(ampl);
    state.get()[row_size * i + k + 4] = std::imag(ampl);
  }

  static void SetEntry(Unitary& state, uint64_t i, uint64_t j, fp_type re,
                       fp_type im) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    uint64_t k = (8 * (j / 4)) + (j % 4);
    state.get()[row_size * i + k] = re;
    state.get()[row_size * i + k + 4] = im;
  }
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_SSE_H_
