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
#include <functional>

#include "unitaryspace.h"
#include "util.h"

namespace qsim {

namespace unitary {

/**
 * Object containing context and routines for unitary manipulations.
 * Unitary is a non-vectorized sequence of one real amplitude followed by
 * one imaginary amplitude.
 */
template <typename For, typename FP>
struct UnitarySpaceBasic
    : public UnitarySpace<UnitarySpaceBasic<For, FP>, For, FP> {
  using Base = UnitarySpace<UnitarySpaceBasic<For, FP>, For, FP>;
  using Unitary = typename Base::Unitary;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit UnitarySpaceBasic(unsigned num_qubits, ForArgs&&... args)
      : Base(2 * (uint64_t{1} << (2 * num_qubits)), num_qubits, args...) {}

  void SetAllZeros(Unitary& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, Unitary& state) {
      state.get()[2 * i + 0] = 0;
      state.get()[2 * i + 1] = 0;
    };

    Base::for_.Run(Base::raw_size_ / 2, f, state);
  }

  void SetIdentity(Unitary& state) {
    auto f = [](unsigned n, unsigned m, uint64_t i, Unitary& state,
                uint64_t dim) {
      auto data = state.get();
      for (uint64_t j = 0; j < dim; j++) {
        data[2 * i * dim + 2 * j] = 0;
        data[2 * i * dim + 2 * j + 1] = 0;
      }
      data[2 * i * dim + 2 * i] = 1;
    };
    Base::for_.Run(Base::Size(), f, state, Base::Size());
  }

  std::complex<fp_type> GetEntry(const Unitary& state, uint64_t i, uint64_t j) {
    uint64_t dim = Base::Size();
    return std::complex<fp_type>(state.get()[2 * i * dim + 2 * j],
                                 state.get()[2 * i * dim + 2 * j + 1]);
  }

  void SetEntry(const Unitary& state, uint64_t i, uint64_t j,
                const std::complex<fp_type>& ampl) {
    uint64_t dim = Base::Size();
    state.get()[2 * i * dim + 2 * j] = std::real(ampl);
    state.get()[2 * i * dim + 2 * j + 1] = std::imag(ampl);
  }

  void SetEntry(const Unitary& state, uint64_t i, uint64_t j, fp_type re,
                fp_type im) {
    uint64_t dim = Base::Size();
    state.get()[2 * i * dim + 2 * j] = re;
    state.get()[2 * i * dim + 2 * j + 1] = im;
  }
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_BASIC_H_f
