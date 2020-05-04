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

#ifndef STATESPACE_SSE_H_
#define STATESPACE_SSE_H_

#include <xmmintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#include "statespace.h"
#include "util.h"

namespace qsim {

// Routines for state-vector manipulations.
// State is a vectorized sequence of four real components followed by four
// imaginary components. Four single-precison floating numbers can be loaded
// into an SSE register.
template <typename ParallelFor>
struct StateSpaceSSE : public StateSpace<ParallelFor, float> {
  using Base = StateSpace<ParallelFor, float>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  StateSpaceSSE(unsigned num_qubits, unsigned num_threads)
      : Base(num_qubits, num_threads,
             2 * std::max(uint64_t{4}, uint64_t{1} << num_qubits)),
        num_qubits_(num_qubits) {}

  void SetAllZeros(State& state) const {
    __m128 val0 = _mm_setzero_ps();

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m128& val0, State& state) {
      _mm_store_ps(state.get() + 8 * i + 0, val0);
      _mm_store_ps(state.get() + 8 * i + 4, val0);
    };

    ParallelFor::Run(Base::num_threads_, Base::raw_size_ / 8, f, val0, state);
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    uint64_t size = uint64_t{1} << num_qubits_;

    __m128 val0 = _mm_setzero_ps();
    __m128 valu = _mm_set1_ps(fp_type{1} / std::sqrt(size));

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m128& val0, const __m128& valu, State& state) {
      _mm_store_ps(state.get() + 8 * i + 0, valu);
      _mm_store_ps(state.get() + 8 * i + 4, val0);
    };

    ParallelFor::Run(Base::num_threads_, Base::raw_size_ / 8, f,
                     val0, valu, state);
  }

  // |0> state.
  void SetStateZero(State& state) const {
    SetAllZeros(state);
    state.get()[0] = 1;
  }

  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    uint64_t p = (8 * (i / 4)) + (i % 4);
    return std::complex<fp_type>(state.get()[p], state.get()[p + 4]);
  }

  static void SetAmpl(
      const State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = (8 * (i / 4)) + (i % 4);
    state.get()[p + 0] = std::real(ampl);
    state.get()[p + 4] = std::imag(ampl);
  }

  static void SetAmpl(const State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = (8 * (i / 4)) + (i % 4);
    state.get()[p + 0] = re;
    state.get()[p + 4] = im;
  }

  double Norm(const State& state) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& state) -> double {
      __m128 re = _mm_load_ps(state.get() + 8 * i);
      __m128 im = _mm_load_ps(state.get() + 8 * i + 4);
      __m128 s1 = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));

      float buffer[4];
      _mm_storeu_ps(buffer, s1);
      return buffer[0] + buffer[1] + buffer[2] + buffer[3];
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::raw_size_ / 8, f, Op(), state);
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    using Op = std::plus<std::complex<double>>;

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) -> std::complex<double> {
      __m128 re1 = _mm_load_ps(state1.get() + 8 * i);
      __m128 im1 = _mm_load_ps(state1.get() + 8 * i + 4);
      __m128 re2 = _mm_load_ps(state2.get() + 8 * i);
      __m128 im2 = _mm_load_ps(state2.get() + 8 * i + 4);

      __m128 ip_re = _mm_add_ps(_mm_mul_ps(re1, re2), _mm_mul_ps(im1, im2));
      __m128 ip_im = _mm_sub_ps(_mm_mul_ps(re1, im2), _mm_mul_ps(im1, re2));

      float bre[4];
      float bim[4];
      _mm_storeu_ps(bre, ip_re);
      _mm_storeu_ps(bim, ip_im);

      double re = bre[0] + bre[1] + bre[2] + bre[3];
      double im = bim[0] + bim[1] + bim[2] + bim[3];

      return std::complex<double>{re, im};
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::raw_size_ / 8, f, Op(), state1, state2);
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) -> double {
      __m128 re1 = _mm_load_ps(state1.get() + 8 * i);
      __m128 im1 = _mm_load_ps(state1.get() + 8 * i + 4);
      __m128 re2 = _mm_load_ps(state2.get() + 8 * i);
      __m128 im2 = _mm_load_ps(state2.get() + 8 * i + 4);

      __m128 ip_re = _mm_add_ps(_mm_mul_ps(re1, re2), _mm_mul_ps(im1, im2));

      float bre[4];
      _mm_storeu_ps(bre, ip_re);

      return bre[0] + bre[1] + bre[2] + bre[3];
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::raw_size_ / 8, f, Op(), state1, state2);
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = 0;
      uint64_t size = Base::raw_size_ / 8;
      const float* v = state.get();

      for (uint64_t k = 0; k < size; ++k) {
        for (unsigned j = 0; j < 4; ++j) {
          auto re = v[8 * k + j];
          auto im = v[8 * k + 4 + j];
          norm += re * re + im * im;
        }
      }

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; ++k) {
        for (unsigned j = 0; j < 4; ++j) {
          auto re = v[8 * k + j];
          auto im = v[8 * k + 4 + j];
          csum += re * re + im * im;
          while (rs[m] < csum && m < num_samples) {
            bitstrings.emplace_back(4 * k + j);
            ++m;
          }
        }
      }
    }

    return bitstrings;
  }

 private:
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // STATESPACE_SSE_H_
