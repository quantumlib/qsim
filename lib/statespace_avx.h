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

#ifndef STATESPACE_AVX_H_
#define STATESPACE_AVX_H_

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#include "statespace.h"

namespace qsim {

// Routines for state-vector manipulations.
// State is a vectorized sequence of eight real amplitudes followed by eight
// imaginary amplitudes. Eight single-precison floating numbers can be loaded
// into an AVX register.
template <typename ParallelFor>
struct StateSpaceAVX final : public StateSpace<ParallelFor, float> {
  using Base = StateSpace<ParallelFor, float>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  StateSpaceAVX(unsigned num_qubits, unsigned num_threads)
      : Base(num_qubits, num_threads,
             2 * std::max(uint64_t{8}, uint64_t{1} << num_qubits)),
        num_qubits_(num_qubits) {}

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    uint64_t size = uint64_t{1} << num_qubits_;

    __m256 val0 = _mm256_setzero_ps();
    __m256 valu = _mm256_set1_ps(fp_type{1} / std::sqrt(size));

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m256& val0, const __m256& valu, State& state) {
      _mm256_store_ps(state.get() + 16 * i + 0, valu);
      _mm256_store_ps(state.get() + 16 * i + 8, val0);
    };

    ParallelFor::Run(Base::num_threads_, Base::raw_size_ / 16, f,
                     val0, valu, state);
  }

  // |0> state.
  void SetStateZero(State& state) const {
    __m256 val0 = _mm256_setzero_ps();

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m256& val0, State& state) {
      _mm256_store_ps(state.get() + 16 * i + 0, val0);
      _mm256_store_ps(state.get() + 16 * i + 8, val0);
    };

    ParallelFor::Run(Base::num_threads_, Base::raw_size_ / 16, f, val0, state);

    state.get()[0] = 1;
  }

  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    uint64_t p = (16 * (i / 8)) + (i % 8);
    return std::complex<fp_type>(state.get()[p], state.get()[p + 8]);
  }

  static void SetAmpl(
      const State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = (16 * (i / 8)) + (i % 8);
    state.get()[p + 0] = std::real(ampl);
    state.get()[p + 8] = std::imag(ampl);
  }

  static void SetAmpl(const State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = (16 * (i / 8)) + (i % 8);
    state.get()[p + 0] = re;
    state.get()[p + 8] = im;
  }

  double Norm(const State& state) const {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& state) -> double {
      __m256 re = _mm256_load_ps(state.get() + 16 * i);
      __m256 im = _mm256_load_ps(state.get() + 16 * i + 8);

      __m256 s1 = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));
      s1 = _mm256_hadd_ps(s1, s1);
      s1 = _mm256_hadd_ps(s1, s1);
      s1 = _mm256_add_ps(s1, _mm256_permute2f128_ps(s1, s1, 1));

      return _mm256_cvtss_f32(s1);
    };

    return ParallelFor::RunReduce(
        Base::num_threads_, Base::raw_size_ / 16, f, Op(), state);
  }

 private:
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // STATESPACE_AVX_H_
