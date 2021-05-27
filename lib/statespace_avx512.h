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

#ifndef STATESPACE_AVX512_H_
#define STATESPACE_AVX512_H_

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#include "statespace.h"
#include "util.h"
#include "vectorspace.h"

namespace qsim {

namespace detail {

inline unsigned GetZeroMaskAVX512(uint64_t i, uint64_t mask, uint64_t bits) {
  __m512i s1 = _mm512_setr_epi64(
      i + 0, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
  __m512i s2 = _mm512_setr_epi64(
      i + 8, i + 9, i + 10, i + 11, i + 12, i + 13, i + 14, i + 15);
  __m512i ma = _mm512_set1_epi64(mask);
  __m512i bi = _mm512_set1_epi64(bits);

  s1 = _mm512_and_si512(s1, ma);
  s2 = _mm512_and_si512(s2, ma);

  unsigned m1 = _mm512_cmpeq_epu64_mask(s1, bi);
  unsigned m2 = _mm512_cmpeq_epu64_mask(s2, bi);

  return (m2 << 8) | m1;
}

inline double HorizontalSumAVX(__m256 s) {
  __m128 l = _mm256_castps256_ps128(s);
  __m128 h = _mm256_extractf128_ps(s, 1);
  __m128 s1  = _mm_add_ps(h, l);
  __m128 s1s = _mm_movehdup_ps(s1);
  __m128 s2 = _mm_add_ps(s1, s1s);

  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehl_ps(s1s, s2)));
}

inline double HorizontalSumAVX512(__m512 s) {
  __m256 l = _mm512_castps512_ps256(s);
  __m512d sd = _mm512_castps_pd(s);
  __m256d hd = _mm512_extractf64x4_pd(sd, 1);
  __m256 h = _mm256_castpd_ps(hd);
  __m256 p = _mm256_add_ps(h, l);

  return HorizontalSumAVX(p);
}

}  // namespace detail

/**
 * Object containing context and routines for AVX state-vector manipulations.
 * State is a vectorized sequence of sixteen real components followed by
 * sixteen imaginary components. Sixteen single-precison floating numbers can
 * be loaded into an AVX512 register.
 */
template <typename For>
class StateSpaceAVX512 :
    public StateSpace<StateSpaceAVX512<For>, VectorSpace, For, float> {
 private:
  using Base = StateSpace<StateSpaceAVX512<For>, qsim::VectorSpace, For, float>;

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit StateSpaceAVX512(ForArgs&&... args) : Base(args...) {}

  static uint64_t MinSize(unsigned num_qubits) {
    return std::max(uint64_t{32}, 2 * (uint64_t{1} << num_qubits));
  };

  void InternalToNormalOrder(State& state) const {
    __m512i idx1 = _mm512_setr_epi32(
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m512i idx2 = _mm512_setr_epi32(
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                __m512i idx1, __m512i idx2, fp_type* p) {
      __m512 v1 = _mm512_load_ps(p + 32 * i);
      __m512 v2 = _mm512_load_ps(p + 32 * i + 16);

      _mm512_store_ps(p + 32 * i,  _mm512_permutex2var_ps(v1, idx1, v2));
      _mm512_store_ps(p + 32 * i + 16,  _mm512_permutex2var_ps(v1, idx2, v2));
    };

    Base::for_.Run(
        MinSize(state.num_qubits()) / 32, f, idx1, idx2, state.get());
  }

  void NormalToInternalOrder(State& state) const {
    __m512i idx1 = _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    __m512i idx2 = _mm512_setr_epi32(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                __m512i idx1, __m512i idx2, fp_type* p) {
      __m512 re = _mm512_load_ps(p + 32 * i);
      __m512 im = _mm512_load_ps(p + 32 * i + 16);

      _mm512_store_ps(p + 32 * i,  _mm512_permutex2var_ps(re, idx1, im));
      _mm512_store_ps(p + 32 * i + 16,  _mm512_permutex2var_ps(re, idx2, im));
    };

    Base::for_.Run(
        MinSize(state.num_qubits()) / 32, f, idx1, idx2, state.get());
  }

  void SetAllZeros(State& state) const {
    __m512 val0 = _mm512_setzero_ps();

    auto f = [](unsigned n, unsigned m, uint64_t i, __m512 val0, fp_type* p) {
      _mm512_store_ps(p + 32 * i, val0);
      _mm512_store_ps(p + 32 * i + 16, val0);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 32, f, val0, state.get());
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    __m512 val0 = _mm512_setzero_ps();
    __m512 valu;

    fp_type v = double{1} / std::sqrt(uint64_t{1} << state.num_qubits());

    switch (state.num_qubits()) {
    case 1:
      valu = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, v, v);
      break;
    case 2:
      valu = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, v, v, v, v);
      break;
    case 3:
      valu = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, v, v, v, v, v, v, v, v);
      break;
    default:
      valu = _mm512_set1_ps(v);
      break;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m512& val0, const __m512& valu, fp_type* p) {
      _mm512_store_ps(p + 32 * i, valu);
      _mm512_store_ps(p + 32 * i + 16, val0);
    };

    Base::for_.Run(
        MinSize(state.num_qubits()) / 32, f, val0, valu, state.get());
  }

  // |0> state.
  void SetStateZero(State& state) const {
    SetAllZeros(state);
    state.get()[0] = 1;
  }

  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    uint64_t p = (32 * (i / 16)) + (i % 16);
    return std::complex<fp_type>(state.get()[p], state.get()[p + 16]);
  }

  static void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = (32 * (i / 16)) + (i % 16);
    state.get()[p] = std::real(ampl);
    state.get()[p + 16] = std::imag(ampl);
  }

  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = (32 * (i / 16)) + (i % 16);
    state.get()[p] = re;
    state.get()[p + 16] = im;
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val,
                   bool exclude = false) const {
    BulkSetAmpl(state, mask, bits, std::real(val), std::imag(val), exclude);
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im, bool exclude = false) const {
    __m512 re_reg = _mm512_set1_ps(re);
    __m512 im_reg = _mm512_set1_ps(im);

    __mmask16 exclude_n = exclude ? 0xffff : 0;

    auto f = [](unsigned n, unsigned m, uint64_t i, uint64_t maskv,
                uint64_t bitsv, __m512 re_n, __m512 im_n, __mmask16 exclude_n,
                fp_type* p) {
      __m512 re = _mm512_load_ps(p + 32 * i);
      __m512 im = _mm512_load_ps(p + 32 * i + 16);

      __mmask16 ml =
          detail::GetZeroMaskAVX512(16 * i, maskv, bitsv) ^ exclude_n;

      re = _mm512_mask_blend_ps(ml, re, re_n);
      im = _mm512_mask_blend_ps(ml, im, im_n);

      _mm512_store_ps(p + 32 * i, re);
      _mm512_store_ps(p + 32 * i + 16, im);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 32, f, mask, bits,
                   re_reg, im_reg, exclude_n, state.get());
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, fp_type* p2) {
      __m512 re1 = _mm512_load_ps(p1 + 32 * i);
      __m512 im1 = _mm512_load_ps(p1 + 32 * i + 16);
      __m512 re2 = _mm512_load_ps(p2 + 32 * i);
      __m512 im2 = _mm512_load_ps(p2 + 32 * i + 16);

      _mm512_store_ps(p2 + 32 * i, _mm512_add_ps(re1, re2));
      _mm512_store_ps(p2 + 32 * i + 16, _mm512_add_ps(im1, im2));
    };

    Base::for_.Run(MinSize(src.num_qubits()) / 32, f, src.get(), dest.get());

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  void Multiply(fp_type a, State& state) const {
    __m512 r = _mm512_set1_ps(a);

    auto f = [](unsigned n, unsigned m, uint64_t i, __m512 r, fp_type* p) {
      __m512 re = _mm512_load_ps(p + 32 * i);
      __m512 im = _mm512_load_ps(p + 32 * i + 16);

      _mm512_store_ps(p + 32 * i, _mm512_mul_ps(re, r));
      _mm512_store_ps(p + 32 * i + 16, _mm512_mul_ps(im, r));
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 32, f, r, state.get());
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> std::complex<double> {
      __m512 re1 = _mm512_load_ps(p1 + 32 * i);
      __m512 im1 = _mm512_load_ps(p1 + 32 * i + 16);
      __m512 re2 = _mm512_load_ps(p2 + 32 * i);
      __m512 im2 = _mm512_load_ps(p2 + 32 * i + 16);

      __m512 ip_re = _mm512_fmadd_ps(im1, im2, _mm512_mul_ps(re1, re2));
      __m512 ip_im = _mm512_fnmadd_ps(im1, re2, _mm512_mul_ps(re1, im2));

      double re = detail::HorizontalSumAVX512(ip_re);
      double im = detail::HorizontalSumAVX512(ip_im);

      return std::complex<double>{re, im};
    };

    using Op = std::plus<std::complex<double>>;
    return Base::for_.RunReduce(MinSize(state1.num_qubits()) / 32, f,
                                Op(), state1.get(), state2.get());
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> double {
      __m512 re1 = _mm512_load_ps(p1 + 32 * i);
      __m512 im1 = _mm512_load_ps(p1 + 32 * i + 16);
      __m512 re2 = _mm512_load_ps(p2 + 32 * i);
      __m512 im2 = _mm512_load_ps(p2 + 32 * i + 16);

      __m512 ip_re = _mm512_fmadd_ps(im1, im2, _mm512_mul_ps(re1, re2));

      return detail::HorizontalSumAVX512(ip_re);
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduce(MinSize(state1.num_qubits()) / 32, f,
                                Op(), state1.get(), state2.get());
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = 0;
      uint64_t size = MinSize(state.num_qubits()) / 32;
      const fp_type* p = state.get();

      for (uint64_t k = 0; k < size; ++k) {
        for (unsigned j = 0; j < 16; ++j) {
          auto re = p[32 * k + j];
          auto im = p[32 * k + 16 + j];
          norm += re * re + im * im;
        }
      }

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; ++k) {
        for (unsigned j = 0; j < 16; ++j) {
          auto re = p[32 * k + j];
          auto im = p[32 * k + 16 + j];
          csum += re * re + im * im;
          while (rs[m] < csum && m < num_samples) {
            bitstrings.emplace_back(16 * k + j);
            ++m;
          }
        }
      }
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  void Collapse(const MeasurementResult& mr, State& state) const {
    auto f1 = [](unsigned n, unsigned m, uint64_t i,
                 uint64_t mask, uint64_t bits, const fp_type* p) -> double {
      __mmask16 ml = detail::GetZeroMaskAVX512(16 * i, mask, bits);

      __m512 re = _mm512_maskz_load_ps(ml, p + 32 * i);
      __m512 im = _mm512_maskz_load_ps(ml, p + 32 * i + 16);
      __m512 s1 = _mm512_fmadd_ps(im, im, _mm512_mul_ps(re, re));

      return detail::HorizontalSumAVX512(s1);
    };

    using Op = std::plus<double>;
    double norm = Base::for_.RunReduce(MinSize(state.num_qubits()) / 32, f1,
                                       Op(), mr.mask, mr.bits, state.get());

    __m512 renorm = _mm512_set1_ps(1.0 / std::sqrt(norm));

    auto f2 = [](unsigned n, unsigned m, uint64_t i,
                 uint64_t mask, uint64_t bits, __m512 renorm, fp_type* p) {
      __mmask16 ml = detail::GetZeroMaskAVX512(16 * i, mask, bits);

      __m512 re = _mm512_maskz_load_ps(ml, p + 32 * i);
      __m512 im = _mm512_maskz_load_ps(ml, p + 32 * i + 16);

      re = _mm512_mul_ps(re, renorm);
      im = _mm512_mul_ps(im, renorm);

      _mm512_store_ps(p + 32 * i, re);
      _mm512_store_ps(p + 32 * i + 16, im);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 32, f2,
                   mr.mask, mr.bits, renorm, state.get());
  }

  std::vector<double> PartialNorms(const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p) -> double {
      __m512 re = _mm512_load_ps(p + 32 * i);
      __m512 im = _mm512_load_ps(p + 32 * i + 16);
      __m512 s1 = _mm512_fmadd_ps(im, im, _mm512_mul_ps(re, re));

      return detail::HorizontalSumAVX512(s1);
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduceP(
        MinSize(state.num_qubits()) / 32, f, Op(), state.get());
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    double csum = 0;

    uint64_t k0 = Base::for_.GetIndex0(MinSize(state.num_qubits()) / 32, m);
    uint64_t k1 = Base::for_.GetIndex1(MinSize(state.num_qubits()) / 32, m);

    const fp_type* p = state.get();

    for (uint64_t k = k0; k < k1; ++k) {
      for (uint64_t j = 0; j < 16; ++j) {
        auto re = p[32 * k + j];
        auto im = p[32 * k + j + 16];
        csum += re * re + im * im;
        if (r < csum) {
          return (16 * k + j) & mask;
        }
      }
    }

    // Return the last bitstring in the unlikely case of underflow.
    return (16 * k1 - 1) & mask;
  }
};

}  // namespace qsim

#endif  // STATESPACE_AVX512_H_
