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

#ifndef UNITARYSPACE_BASIC_H_
#define UNITARYSPACE_BASIC_H_

#include <cmath>
#include <complex>
#include <cstdint>

#include "unitaryspace.h"
#include "vectorspace.h"

namespace qsim {

namespace unitary {

/**
 * Object containing context and routines for unitary manipulations.
 * Unitary is a non-vectorized sequence of one real amplitude followed by
 * one imaginary amplitude.
 */
template <typename For, typename FP>
struct UnitarySpaceBasic
    : public UnitarySpace<UnitarySpaceBasic<For, FP>, VectorSpace, For, FP> {
 private:
  using Base = UnitarySpace<UnitarySpaceBasic<For, FP>,
                            qsim::VectorSpace, For, FP>;

 public:
  using Unitary = typename Base::Unitary;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit UnitarySpaceBasic(ForArgs&&... args) : Base(args...) {}

  static uint64_t MinRowSize(unsigned num_qubits) {
    return 2 * (uint64_t{1} << num_qubits);
  };

  static uint64_t MinSize(unsigned num_qubits) {
    return Base::Size(num_qubits) * MinRowSize(num_qubits);
  };

  void SetAllZeros(Unitary& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, fp_type* p) {
      p[2 * i + 0] = 0;
      p[2 * i + 1] = 0;
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f, state.get());
  }

  void SetIdentity(Unitary& state) {
    SetAllZeros(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t row_size, fp_type* p) {
      p[row_size * i + 2 * i] = 1;
    };

    uint64_t size = Base::Size(state.num_qubits());
    uint64_t row_size = MinRowSize(state.num_qubits());
    Base::for_.Run(size, f, row_size, state.get());
  }

  static std::complex<fp_type> GetEntry(const Unitary& state,
                                        uint64_t i, uint64_t j) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    return std::complex<fp_type>(state.get()[row_size * i + 2 * j],
                                 state.get()[row_size * i + 2 * j + 1]);
  }

  static void SetEntry(Unitary& state, uint64_t i, uint64_t j,
                       const std::complex<fp_type>& ampl) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    state.get()[row_size * i + 2 * j] = std::real(ampl);
    state.get()[row_size * i + 2 * j + 1] = std::imag(ampl);
  }

  static void SetEntry(Unitary& state, uint64_t i, uint64_t j,
                       fp_type re, fp_type im) {
    uint64_t row_size = MinRowSize(state.num_qubits());
    state.get()[row_size * i + 2 * j] = re;
    state.get()[row_size * i + 2 * j + 1] = im;
  }
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_BASIC_H_
