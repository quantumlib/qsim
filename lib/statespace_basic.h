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

#include "statespace.h"
#include "util.h"

namespace qsim {

// Routines for state-vector manipulations.
// State is a non-vectorized sequence of one real amplitude followed by
// one imaginary amplitude.
template <typename ParallelFor, typename FP>
struct StateSpaceBasic : public StateSpace<ParallelFor, FP> {
  using Base = StateSpace<ParallelFor, FP>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  StateSpaceBasic(unsigned num_qubits, unsigned num_threads)
      : Base(num_qubits, num_threads, 2 * (uint64_t{1} << num_qubits)) {}

  void SetAllZeros(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state) {
      state.get()[2 * i + 0] = 0;
      state.get()[2 * i + 1] = 0;
    };

    ParallelFor::Run(Base::num_threads_, Base::size_, f, state);
  }

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
    SetAllZeros(state);
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

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    using Op = std::plus<std::complex<double>>;

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) -> std::complex<double> {
      auto s1 = state1.get() + 2 * i;
      auto s2 = state2.get() + 2 * i;

      double re = s1[0] * s2[0] + s1[1] * s2[1];
      double im = s1[0] * s2[1] - s1[1] * s2[0];

      return std::complex<double>{re, im};
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::size_, f, Op(), state1, state2);
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) -> double {
      auto s1 = state1.get() + 2 * i;
      auto s2 = state2.get() + 2 * i;

      return s1[0] * s2[0] + s1[1] * s2[1];
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::size_, f, Op(), state1, state2);
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = 0;
      uint64_t size = 2 * Base::size_;
      const fp_type* v = state.get();

      for (uint64_t k = 0; k < size; k += 2) {
        norm += v[k] * v[k] + v[k + 1] * v[k + 1];
      }

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; k += 2) {
        csum += v[k] * v[k] + v[k + 1] * v[k + 1];
        while (rs[m] < csum && m < num_samples) {
          bitstrings.emplace_back(k / 2);
          ++m;
        }
      }
    }

    return bitstrings;
  }
};

}  // namespace qsim

#endif  // STATESPACE_BASIC_H_f
