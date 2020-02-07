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

#ifndef STATESPACE_BASIC_H_
#define STATESPACE_BASIC_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#include "parfor.h"
#include "statespace.h"

namespace qsim {

// Routines for state-vector manipulations.
// State is a non-vectorized sequence of one real amplitude followed by
// one imaginary amplitude.
template <typename ParallelFor, typename FP>
struct StateSpaceBasic final : public StateSpace<ParallelFor, FP> {
  using Base = StateSpace<ParallelFor, FP>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  StateSpaceBasic(unsigned num_qubits, unsigned num_threads)
      : Base(num_qubits, num_threads, 2 * (uint64_t{1} << num_qubits)) {}

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    fp_type val = fp_type{1} / std::sqrt(Base::size_);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                fp_type val, State& state) {
      state.get()[2 * i + 0] = val;
      state.get()[2 * i + 1] = 0;
    };

    ParallelFor::Run(
        Base::num_threads_, Base::size_, f, state, val, state);
  }

  // |0> state.
  void SetStateZero(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state) {
      state.get()[2 * i + 0] = 0;
      state.get()[2 * i + 1] = 0;
    };

    ParallelFor::Run(Base::num_threads_, Base::size_, f, state);

    state.get()[0] = 1;
  }

  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    uint64_t p = 2 * i;
    return std::complex<fp_type>(state.get()[p], state.get()[p + 1]);
  }

  static void SetAmpl(
      const State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = 2 * i;
    state.get()[p + 0] = std::real(ampl);
    state.get()[p + 1] = std::imag(ampl);
  }

  static void SetAmpl(const State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = 2 * i;
    state.get()[p + 0] = re;
    state.get()[p + 1] = im;
  }

  double Norm(const State& state) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& state) -> double {
      auto s = state.get() + 2 * i;
      return s[0] * s[0] + s[1] * s[1];
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::size_, f, Op(), state);
  }
};

}  // namespace qsim

#endif  // STATESPACE_BASIC_H_f
