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

/**
 * Object containing context and routines for unoptimized state-vector
 * manipulations. State is a non-vectorized sequence of one real amplitude
 * followed by one imaginary amplitude.
 */
template <typename For, typename FP>
class StateSpaceBasic : public StateSpace<StateSpaceBasic<For, FP>, For, FP> {
 private:
  using Base = StateSpace<StateSpaceBasic<For, FP>, For, FP>;

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit StateSpaceBasic(ForArgs&&... args) : Base(args...) {}

  static uint64_t MinSize(unsigned num_qubits) {
    return 2 * (uint64_t{1} << num_qubits);
  };

  void InternalToNormalOrder(State& state) const {}

  void NormalToInternalOrder(State& state) const {}

  void SetAllZeros(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, fp_type* p) {
      p[2 * i] = 0;
      p[2 * i + 1] = 0;
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f, state.get());
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    fp_type val = fp_type{1} / std::sqrt(uint64_t{1} << state.num_qubits());

    auto f = [](unsigned n, unsigned m, uint64_t i,
                fp_type val, fp_type* p) {
      p[2 * i] = val;
      p[2 * i + 1] = 0;
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f, val, state.get());
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
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = 2 * i;
    state.get()[p] = std::real(ampl);
    state.get()[p + 1] = std::imag(ampl);
  }

  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = 2 * i;
    state.get()[p] = re;
    state.get()[p + 1] = im;
  }

  // Sets state[i] = val where (i & mask) == bits
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val) const {
    BulkSetAmpl(state, mask, bits, std::real(val), std::imag(val));
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, uint64_t maskv,
                uint64_t bitsv, fp_type re_n, fp_type im_n, fp_type* p) {
      auto s = p + 2 * i;
      bool in_mask = (i & maskv) == bitsv;

      s[0] = in_mask ? re_n : s[0];
      s[1] = in_mask ? im_n : s[1];
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f, mask, bits, re, im,
                   state.get());
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, fp_type* p2) {
      p2[2 * i] += p1[2 * i];
      p2[2 * i + 1] += p1[2 * i + 1];
    };

    Base::for_.Run(MinSize(src.num_qubits()) / 2, f, src.get(), dest.get());

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  void Multiply(fp_type a, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, fp_type a, fp_type* p) {
      p[2 * i] *= a;
      p[2 * i + 1] *= a;
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f, a, state.get());
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> std::complex<double> {
      auto s1 = p1 + 2 * i;
      auto s2 = p2 + 2 * i;

      double re = s1[0] * s2[0] + s1[1] * s2[1];
      double im = s1[0] * s2[1] - s1[1] * s2[0];

      return std::complex<double>{re, im};
    };

    using Op = std::plus<std::complex<double>>;
    return Base::for_.RunReduce(
        MinSize(state1.num_qubits()) / 2, f, Op(), state1.get(), state2.get());
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> double {
      auto s1 = p1 + 2 * i;
      auto s2 = p2 + 2 * i;

      return s1[0] * s2[0] + s1[1] * s2[1];
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduce(
        MinSize(state1.num_qubits()) / 2, f, Op(), state1.get(), state2.get());
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = 0;
      uint64_t size = MinSize(state.num_qubits()) / 2;

      const fp_type* p = state.get();

      for (uint64_t k = 0; k < size; ++k) {
        auto re = p[2 * k];
        auto im = p[2 * k + 1];
        norm += re * re + im * im;
      }

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; ++k) {
        auto re = p[2 * k];
        auto im = p[2 * k + 1];
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

  void Collapse(const MeasurementResult& mr, State& state) const {
    auto f1 = [](unsigned n, unsigned m, uint64_t i,
                 uint64_t mask, uint64_t bits, const fp_type* p) -> double {
      auto s = p + 2 * i;
      return (i & mask) == bits ? s[0] * s[0] + s[1] * s[1] : 0;
    };

    using Op = std::plus<double>;
    double norm = Base::for_.RunReduce(MinSize(state.num_qubits()) / 2, f1,
                                       Op(), mr.mask, mr.bits, state.get());

    double renorm = 1.0 / std::sqrt(norm);

    auto f2 = [](unsigned n, unsigned m, uint64_t i,
                 uint64_t mask, uint64_t bits, fp_type renorm, fp_type* p) {
      auto s = p + 2 * i;
      bool not_zero = (i & mask) == bits;

      s[0] = not_zero ? s[0] * renorm : 0;
      s[1] = not_zero ? s[1] * renorm : 0;
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 2, f2,
                   mr.mask, mr.bits, renorm, state.get());
  }

  std::vector<double> PartialNorms(const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p) -> double {
      auto s = p + 2 * i;
      return s[0] * s[0] + s[1] * s[1];
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduceP(
        MinSize(state.num_qubits()) / 2, f, Op(), state.get());
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    double csum = 0;

    uint64_t k0 = Base::for_.GetIndex0(MinSize(state.num_qubits()) / 2, m);
    uint64_t k1 = Base::for_.GetIndex1(MinSize(state.num_qubits()) / 2, m);

    const fp_type* p = state.get();

    for (uint64_t k = k0; k < k1; ++k) {
      auto re = p[2 * k];
      auto im = p[2 * k + 1];
      csum += re * re + im * im;
      if (r < csum) {
        return k & mask;
      }
    }

    return -1;
  }
};

}  // namespace qsim

#endif  // STATESPACE_BASIC_H_f
