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

#include <smmintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#include "statespace.h"
#include "util.h"

namespace qsim {

namespace detail {

inline __m128i GetZeroMaskSSE(uint64_t i, uint64_t mask, uint64_t bits) {
  __m128i s1 = _mm_set_epi64x(i + 2, i + 0);
  __m128i s2 = _mm_set_epi64x(i + 3, i + 1);
  __m128i ma = _mm_set1_epi64x(mask);
  __m128i bi = _mm_set1_epi64x(bits);

  s1 = _mm_and_si128(s1, ma);
  s2 = _mm_and_si128(s2, ma);

  s1 = _mm_cmpeq_epi64(s1, bi);
  s2 = _mm_cmpeq_epi64(s2, bi);

  return _mm_blend_epi16(s1, s2, 204);  // 11001100
}

inline double HorizontalSumSSE(__m128 s) {
  float buf[4];
  _mm_storeu_ps(buf, s);
  return buf[0] + buf[1] + buf[2] + buf[3];
}

}  // namespace detail

// Routines for state-vector manipulations.
// State is a vectorized sequence of four real components followed by four
// imaginary components. Four single-precison floating numbers can be loaded
// into an SSE register.
template <typename For>
struct StateSpaceSSE : public StateSpace<StateSpaceSSE<For>, For, float> {
  using Base = StateSpace<StateSpaceSSE<For>, For, float>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit StateSpaceSSE(unsigned num_qubits, ForArgs&&... args)
      : Base(2 * std::max(uint64_t{4}, uint64_t{1} << num_qubits),
             num_qubits, args...) {}

  void InternalToNormalOrder(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state) {
      auto s = state.get() + 8 * i;

      fp_type re[3];
      fp_type im[3];

      for (uint64_t i = 0; i < 3; ++i) {
        re[i] = s[i + 1];
        im[i] = s[i + 4];
      }

      for (uint64_t i = 0; i < 3; ++i) {
        s[2 * i + 1] = im[i];
        s[2 * i + 2] = re[i];
      }
    };

    if (Base::num_qubits_ == 1) {
      auto s = state.get();

      s[2] = s[1];
      s[1] = s[4];
      s[3] = s[5];

      for (uint64_t i = 4; i < 8; ++i) {
        s[i] = 0;
      }
    } else {
      Base::for_.Run(Base::raw_size_ / 8, f, state);
    }
  }

  void NormalToInternalOrder(State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, State& state) {
      auto s = state.get() + 8 * i;

      fp_type re[3];
      fp_type im[3];

      for (uint64_t i = 0; i < 3; ++i) {
        im[i] = s[2 * i + 1];
        re[i] = s[2 * i + 2];
      }

      for (uint64_t i = 0; i < 3; ++i) {
        s[i + 1] = re[i];
        s[i + 4] = im[i];
      }
    };

    if (Base::num_qubits_ == 1) {
      auto s = state.get();

      s[4] = s[1];
      s[1] = s[2];
      s[5] = s[3];

      s[2] = 0;
      s[3] = 0;
      s[6] = 0;
      s[7] = 0;
    } else {
      Base::for_.Run(Base::raw_size_ / 8, f, state);
    }
  }

  void SetAllZeros(State& state) const {
    __m128 val0 = _mm_setzero_ps();

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m128& val0, State& state) {
      _mm_store_ps(state.get() + 8 * i + 0, val0);
      _mm_store_ps(state.get() + 8 * i + 4, val0);
    };

    Base::for_.Run(Base::raw_size_ / 8, f, val0, state);
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    __m128 val0 = _mm_setzero_ps();
    __m128 valu;

    fp_type v = double{1} / std::sqrt(Base::Size());

    if (Base::num_qubits_ == 1) {
      valu = _mm_set_ps(0, 0, v, v);
    } else {
      valu = _mm_set1_ps(v);
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m128& val0, const __m128& valu, State& state) {
      _mm_store_ps(state.get() + 8 * i + 0, valu);
      _mm_store_ps(state.get() + 8 * i + 4, val0);
    };

    Base::for_.Run(Base::raw_size_ / 8, f, val0, valu, state);
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

  // Does the equivalent of dest += source elementwise.
  void AddState(const State& source, const State& dest) const {

    auto f = [](unsigned n, unsigned m, uint64_t i, const State& state1,
                const State& state2) {
      __m128 re1 = _mm_load_ps(state1.get() + 8 * i);
      __m128 im1 = _mm_load_ps(state1.get() + 8 * i + 4);
      __m128 re2 = _mm_load_ps(state2.get() + 8 * i);
      __m128 im2 = _mm_load_ps(state2.get() + 8 * i + 4);

      _mm_store_ps(state2.get() + 8 * i, _mm_add_ps(re1, re2));
      _mm_store_ps(state2.get() + 8 * i + 4, _mm_add_ps(im1, im2));
    };

    Base::for_.Run(Base::raw_size_ / 8, f, source, dest);
  }

  void Multiply(fp_type a, State& state) const {
    __m128 r = _mm_set1_ps(a);

    auto f = [](unsigned n, unsigned m, uint64_t i, State& state, __m128 r) {
      __m128 re = _mm_load_ps(state.get() + 8 * i);
      __m128 im = _mm_load_ps(state.get() + 8 * i + 4);

      re = _mm_mul_ps(re, r);
      im = _mm_mul_ps(im, r);

      _mm_store_ps(state.get() + 8 * i, re);
      _mm_store_ps(state.get() + 8 * i + 4, im);
    };

    Base::for_.Run(Base::raw_size_ / 8, f, state, r);
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

      double re = detail::HorizontalSumSSE(ip_re);
      double im = detail::HorizontalSumSSE(ip_im);

      return std::complex<double>{re, im};
    };

    return Base::for_.RunReduce(Base::raw_size_ / 8, f, Op(), state1, state2);
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

      return detail::HorizontalSumSSE(ip_re);
    };

    return Base::for_.RunReduce(Base::raw_size_ / 8, f, Op(), state1, state2);
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

  using MeasurementResult = typename Base::MeasurementResult;

  void CollapseState(const MeasurementResult& mr, State& state) const {
    __m128 zero = _mm_set1_ps(0);

    auto f1 = [](unsigned n, unsigned m, uint64_t i, const State& state,
                 uint64_t mask, uint64_t bits, __m128 zero) -> double {
      __m128 ml = _mm_castsi128_ps(detail::GetZeroMaskSSE(4 * i, mask, bits));

      __m128 re = _mm_load_ps(state.get() + 8 * i);
      __m128 im = _mm_load_ps(state.get() + 8 * i + 4);
      __m128 s1 = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));

      s1 = _mm_blendv_ps(zero, s1, ml);

      return detail::HorizontalSumSSE(s1);
    };

    using Op = std::plus<double>;
    double norm = Base::for_.RunReduce(Base::raw_size_ / 8, f1,
                                       Op(), state, mr.mask, mr.bits, zero);

    __m128 renorm = _mm_set1_ps(1.0 / std::sqrt(norm));

    auto f2 = [](unsigned n, unsigned m, uint64_t i,  State& state,
                 uint64_t mask, uint64_t bits, __m128 renorm, __m128 zero) {
      __m128 ml = _mm_castsi128_ps(detail::GetZeroMaskSSE(4 * i, mask, bits));

      __m128 re = _mm_load_ps(state.get() + 8 * i);
      __m128 im = _mm_load_ps(state.get() + 8 * i + 4);

      re = _mm_blendv_ps(zero, _mm_mul_ps(re, renorm), ml);
      im = _mm_blendv_ps(zero, _mm_mul_ps(im, renorm), ml);

      _mm_store_ps(state.get() + 8 * i, re);
      _mm_store_ps(state.get() + 8 * i + 4, im);
    };

    Base::for_.Run(
        Base::raw_size_ / 8, f2, state, mr.mask, mr.bits, renorm, zero);
  }

  std::vector<double> PartialNorms(const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& state) -> double {
      __m128 re = _mm_load_ps(state.get() + 8 * i);
      __m128 im = _mm_load_ps(state.get() + 8 * i + 4);
      __m128 s1 = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));

      return detail::HorizontalSumSSE(s1);
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduceP(Base::raw_size_ / 8, f, Op(), state);
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    double csum = 0;

    uint64_t k0 = Base::for_.GetIndex0(Base::raw_size_ / 8, m);
    uint64_t k1 = Base::for_.GetIndex1(Base::raw_size_ / 8, m);

    for (uint64_t k = k0; k < k1; ++k) {
      for (uint64_t j = 0; j < 4; ++j) {
        auto re = state.get()[8 * k + j];
        auto im = state.get()[8 * k + 4 + j];
        csum += re * re + im * im;
        if (r < csum) {
          return (4 * k + j) & mask;
        }
      }
    }

    return 0;
  }
};

}  // namespace qsim

#endif  // STATESPACE_SSE_H_
