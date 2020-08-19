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
template <typename For, typename FP>
struct StateSpaceBasic : public StateSpace<StateSpaceBasic<For, FP>, For, FP> {
  using Base = StateSpace<StateSpaceBasic<For, FP>, For, FP>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit StateSpaceBasic(unsigned num_qubits, ForArgs&&... args)
      : Base(2 * (uint64_t{1} << num_qubits), num_qubits, args...) {}

  void InternalToNormalOrder(State& state) const {}

  void NormalToInternalOrder(State& state) const {}

  void SetAllZeros(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state) {
      state.get()[2 * i + 0] = 0;
      state.get()[2 * i + 1] = 0;
    };

    Base::for_.Run(Base::raw_size_ / 2, f, state);
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    fp_type val = fp_type{1} / std::sqrt(Base::Size());

    auto f = [](unsigned n, unsigned m, uint64_t i,
                fp_type val, State& state) {
      state.get()[2 * i + 0] = val;
      state.get()[2 * i + 1] = 0;
    };

    Base::for_.Run(Base::raw_size_ / 2, f, val, state);
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

  // Does the equivalent of dest += source elementwise.
  void AddState(const State& source, const State& dest) const {

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) {
      state2.get()[2 * i + 0] += state1.get()[2 * i + 0];
      state2.get()[2 * i + 1] += state1.get()[2 * i + 1];
    };

    Base::for_.Run(Base::raw_size_ / 2, f, source, dest);
  }

  void Multiply(fp_type a, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state, fp_type a) {
      state.get()[2 * i + 0] *= a;
      state.get()[2 * i + 1] *= a;
    };

    Base::for_.Run(Base::raw_size_ / 2, f, state, a);
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

    return Base::for_.RunReduce(Base::raw_size_ / 2, f, Op(), state1, state2);
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) -> double {
      auto s1 = state1.get() + 2 * i;
      auto s2 = state2.get() + 2 * i;

      return s1[0] * s2[0] + s1[1] * s2[1];
    };

    return Base::for_.RunReduce(Base::raw_size_ / 2, f, Op(), state1, state2);
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = 0;
      uint64_t size = Base::raw_size_ / 2;

      for (uint64_t k = 0; k < size; ++k) {
        auto re = state.get()[2 * k];
        auto im = state.get()[2 * k + 1];
        norm += re * re + im * im;
      }

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; ++k) {
        auto re = state.get()[2 * k];
        auto im = state.get()[2 * k + 1];
        csum += re * re + im * im;
        while (rs[m] < csum && m < num_samples) {
          bitstrings.emplace_back(k);
          ++m;
        }
      }
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  void CollapseState(const MeasurementResult& mr, State& state) const {
    auto f1 = [](unsigned n, unsigned m, uint64_t i, const State& state,
                 uint64_t mask, uint64_t bits) -> double {
      auto s = state.get() + 2 * i;
      return (i & mask) == bits ? s[0] * s[0] + s[1] * s[1] : 0;
    };

    using Op = std::plus<double>;
    double norm = Base::for_.RunReduce(Base::raw_size_ / 2, f1,
                                       Op(), state, mr.mask, mr.bits);

    double renorm = 1.0 / std::sqrt(norm);

    auto f2 = [](unsigned n, unsigned m, uint64_t i,  State& state,
                 uint64_t mask, uint64_t bits, fp_type renorm) {
      auto s = state.get() + 2 * i;
      bool not_zero = (i & mask) == bits;

      s[0] = not_zero ? s[0] * renorm : 0;
      s[1] = not_zero ? s[1] * renorm : 0;
    };

    Base::for_.Run(Base::raw_size_ / 2, f2, state, mr.mask, mr.bits, renorm);
  }

  std::vector<double> PartialNorms(const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& state) -> double {
      auto s = state.get() + 2 * i;
      return s[0] * s[0] + s[1] * s[1];
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduceP(Base::raw_size_ / 2, f, Op(), state);
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    double csum = 0;

    uint64_t k0 = Base::for_.GetIndex0(Base::raw_size_ / 2, m);
    uint64_t k1 = Base::for_.GetIndex1(Base::raw_size_ / 2, m);

    for (uint64_t k = k0; k < k1; ++k) {
      auto re = state.get()[2 * k];
      auto im = state.get()[2 * k + 1];
      csum += re * re + im * im;
      if (r < csum) {
        return k & mask;
      }
    }

    return 0;
  }
};

}  // namespace qsim

#endif  // STATESPACE_BASIC_H_f
