// Copyright 2026 Google LLC. All Rights Reserved.
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

#ifndef STATESPACE_NEON_H_
#define STATESPACE_NEON_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>

#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
#error "statespace_neon.h requires __ARM_NEON__."
#endif
#include <arm_neon.h>

#include "statespace.h"
#include "util.h"
#include "vectorspace.h"

namespace qsim {

namespace detail {

inline double HorizontalSumNEON(float32x4_t s) {
  float32x2_t s2 = vadd_f32(vget_low_f32(s), vget_high_f32(s));
  s2 = vpadd_f32(s2, s2);
  return vget_lane_f32(s2, 0);
}

inline uint32x4_t GetZeroMaskNEON(uint64_t i, uint64_t mask, uint64_t bits) {
  alignas(16) uint32_t lanes[4];
  for (unsigned j = 0; j < 4; ++j) {
    lanes[j] = ((i + j) & mask) == bits ? ~uint32_t{0} : 0;
  }
  return vld1q_u32(lanes);
}

}  // namespace detail

/**
 * Object containing context and routines for NEON state-vector manipulations.
 * State is a vectorized sequence of four real components followed by four
 * imaginary components. Four single-precison floating numbers can be loaded
 * into a NEON register.
 */
template <typename For>
class StateSpaceNEON :
    public StateSpace<StateSpaceNEON<For>, VectorSpace, For, float> {
 private:
  using Base = StateSpace<StateSpaceNEON<For>, qsim::VectorSpace, For, float>;

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  template <typename... ForArgs>
  explicit StateSpaceNEON(ForArgs&&... args) : Base(args...) {}

  static uint64_t MinSize(unsigned num_qubits) {
    return std::max(uint64_t{8}, 2 * (uint64_t{1} << num_qubits));
  }

  void InternalToNormalOrder(State& state) const {
    if (state.num_qubits() == 1) {
      auto s = state.get();

      s[2] = s[1];
      s[1] = s[4];
      s[3] = s[5];

      for (uint64_t i = 4; i < 8; ++i) {
        s[i] = 0;
      }
    } else {
      auto f = [](unsigned n, unsigned m, uint64_t i, fp_type* p) {
        auto s = p + 8 * i;

        fp_type re[3];
        fp_type im[3];

        for (uint64_t j = 0; j < 3; ++j) {
          re[j] = s[j + 1];
          im[j] = s[j + 4];
        }

        for (uint64_t j = 0; j < 3; ++j) {
          s[2 * j + 1] = im[j];
          s[2 * j + 2] = re[j];
        }
      };

      Base::for_.Run(MinSize(state.num_qubits()) / 8, f, state.get());
    }
  }

  void NormalToInternalOrder(State& state) const {
    if (state.num_qubits() == 1) {
      auto s = state.get();

      s[4] = s[1];
      s[1] = s[2];
      s[5] = s[3];

      s[2] = 0;
      s[3] = 0;
      s[6] = 0;
      s[7] = 0;
    } else {
      auto f = [](unsigned n, unsigned m, uint64_t i, fp_type* p) {
        auto s = p + 8 * i;

        fp_type re[3];
        fp_type im[3];

        for (uint64_t j = 0; j < 3; ++j) {
          im[j] = s[2 * j + 1];
          re[j] = s[2 * j + 2];
        }

        for (uint64_t j = 0; j < 3; ++j) {
          s[j + 1] = re[j];
          s[j + 4] = im[j];
        }
      };

      Base::for_.Run(MinSize(state.num_qubits()) / 8, f, state.get());
    }
  }

  void SetAllZeros(State& state) const {
    float32x4_t zero = vdupq_n_f32(0.0f);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                float32x4_t zero, fp_type* p) {
      vst1q_f32(p + 8 * i, zero);
      vst1q_f32(p + 8 * i + 4, zero);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f, zero, state.get());
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    fp_type v = fp_type{1} / std::sqrt(uint64_t{1} << state.num_qubits());

    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t valu;

    if (state.num_qubits() == 1) {
      alignas(16) float lanes[4] = {v, v, 0, 0};
      valu = vld1q_f32(lanes);
    } else {
      valu = vdupq_n_f32(v);
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                float32x4_t zero, float32x4_t valu, fp_type* p) {
      vst1q_f32(p + 8 * i, valu);
      vst1q_f32(p + 8 * i + 4, zero);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f, zero, valu, state.get());
  }

  // |0> state.
  void SetStateZero(State& state) const {
    SetAllZeros(state);
    state.get()[0] = 1;
  }

  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    uint64_t p = 8 * (i / 4) + (i % 4);
    return std::complex<fp_type>(state.get()[p], state.get()[p + 4]);
  }

  static void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    uint64_t p = 8 * (i / 4) + (i % 4);
    state.get()[p] = std::real(ampl);
    state.get()[p + 4] = std::imag(ampl);
  }

  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
    uint64_t p = 8 * (i / 4) + (i % 4);
    state.get()[p] = re;
    state.get()[p + 4] = im;
  }

  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val,
                   bool exclude = false) const {
    BulkSetAmpl(state, mask, bits, std::real(val), std::imag(val), exclude);
  }

  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im, bool exclude = false) const {
    float32x4_t re_n = vdupq_n_f32(re);
    float32x4_t im_n = vdupq_n_f32(im);
    uint32x4_t exclude_n = exclude ? vdupq_n_u32(~uint32_t{0}) : vdupq_n_u32(0);

    auto f = [](unsigned n, unsigned m, uint64_t i, uint64_t maskv,
                uint64_t bitsv, float32x4_t re_n, float32x4_t im_n,
                uint32x4_t exclude_n, fp_type* p) {
      uint32x4_t ml = veorq_u32(detail::GetZeroMaskNEON(4 * i, maskv, bitsv),
                                exclude_n);
      float32x4_t re = vld1q_f32(p + 8 * i);
      float32x4_t im = vld1q_f32(p + 8 * i + 4);

      re = vbslq_f32(ml, re_n, re);
      im = vbslq_f32(ml, im_n, im);

      vst1q_f32(p + 8 * i, re);
      vst1q_f32(p + 8 * i + 4, im);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f, mask, bits, re_n,
                   im_n, exclude_n, state.get());
  }

  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) return false;

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, fp_type* p2) {
      float32x4_t re1 = vld1q_f32(p1 + 8 * i);
      float32x4_t im1 = vld1q_f32(p1 + 8 * i + 4);
      float32x4_t re2 = vld1q_f32(p2 + 8 * i);
      float32x4_t im2 = vld1q_f32(p2 + 8 * i + 4);

      vst1q_f32(p2 + 8 * i, vaddq_f32(re1, re2));
      vst1q_f32(p2 + 8 * i + 4, vaddq_f32(im1, im2));
    };

    Base::for_.Run(MinSize(src.num_qubits()) / 8, f, src.get(), dest.get());
    return true;
  }

  void Multiply(fp_type a, State& state) const {
    float32x4_t r = vdupq_n_f32(a);

    auto f = [](unsigned n, unsigned m, uint64_t i, float32x4_t r, fp_type* p) {
      float32x4_t re = vld1q_f32(p + 8 * i);
      float32x4_t im = vld1q_f32(p + 8 * i + 4);

      vst1q_f32(p + 8 * i, vmulq_f32(re, r));
      vst1q_f32(p + 8 * i + 4, vmulq_f32(im, r));
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f, r, state.get());
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> std::complex<double> {
      float32x4_t re1 = vld1q_f32(p1 + 8 * i);
      float32x4_t im1 = vld1q_f32(p1 + 8 * i + 4);
      float32x4_t re2 = vld1q_f32(p2 + 8 * i);
      float32x4_t im2 = vld1q_f32(p2 + 8 * i + 4);

      float32x4_t ip_re = vmulq_f32(re1, re2);
      ip_re = vfmaq_f32(ip_re, im1, im2);
      float32x4_t ip_im = vmulq_f32(re1, im2);
      ip_im = vfmsq_f32(ip_im, im1, re2);

      return std::complex<double>(detail::HorizontalSumNEON(ip_re),
                                  detail::HorizontalSumNEON(ip_im));
    };

    using Op = std::plus<std::complex<double>>;
    return Base::for_.RunReduce(
        MinSize(state1.num_qubits()) / 8, f, Op(), state1.get(), state2.get());
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p1, const fp_type* p2) -> double {
      float32x4_t re1 = vld1q_f32(p1 + 8 * i);
      float32x4_t im1 = vld1q_f32(p1 + 8 * i + 4);
      float32x4_t re2 = vld1q_f32(p2 + 8 * i);
      float32x4_t im2 = vld1q_f32(p2 + 8 * i + 4);

      float32x4_t ip_re = vmulq_f32(re1, re2);
      ip_re = vfmaq_f32(ip_re, im1, im2);
      return detail::HorizontalSumNEON(ip_re);
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduce(
        MinSize(state1.num_qubits()) / 8, f, Op(), state1.get(), state2.get());
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      double norm = this->Norm(state);
      uint64_t size = MinSize(state.num_qubits()) / 8;
      const fp_type* p = state.get();

      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      uint64_t m = 0;
      double csum = 0;
      bitstrings.reserve(num_samples);

      for (uint64_t k = 0; k < size; ++k) {
        for (unsigned j = 0; j < 4; ++j) {
          double re = p[8 * k + j];
          double im = p[8 * k + 4 + j];
          csum += re * re + im * im;
          while (m < num_samples && rs[m] < csum) {
            bitstrings.emplace_back(4 * k + j);
            ++m;
          }
        }
      }

      for (; m < num_samples; ++m) {
        bitstrings.emplace_back((uint64_t{1} << state.num_qubits()) - 1);
      }
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  void Collapse(const MeasurementResult& mr, State& state) const {
    float32x4_t zero = vdupq_n_f32(0.0f);

    auto f1 = [](unsigned n, unsigned m, uint64_t i, uint64_t mask,
                 uint64_t bits, float32x4_t zero,
                 const fp_type* p) -> double {
      uint32x4_t ml = detail::GetZeroMaskNEON(4 * i, mask, bits);
      float32x4_t re = vld1q_f32(p + 8 * i);
      float32x4_t im = vld1q_f32(p + 8 * i + 4);
      float32x4_t s1 = vmulq_f32(re, re);
      s1 = vfmaq_f32(s1, im, im);
      s1 = vbslq_f32(ml, s1, zero);
      return detail::HorizontalSumNEON(s1);
    };

    using Op = std::plus<double>;
    double norm = Base::for_.RunReduce(MinSize(state.num_qubits()) / 8, f1,
                                       Op(), mr.mask, mr.bits, zero,
                                       state.get());

    float32x4_t renorm = vdupq_n_f32(1.0 / std::sqrt(norm));

    auto f2 = [](unsigned n, unsigned m, uint64_t i, uint64_t mask,
                 uint64_t bits, float32x4_t renorm, float32x4_t zero,
                 fp_type* p) {
      uint32x4_t ml = detail::GetZeroMaskNEON(4 * i, mask, bits);
      float32x4_t re = vld1q_f32(p + 8 * i);
      float32x4_t im = vld1q_f32(p + 8 * i + 4);

      re = vbslq_f32(ml, vmulq_f32(re, renorm), zero);
      im = vbslq_f32(ml, vmulq_f32(im, renorm), zero);

      vst1q_f32(p + 8 * i, re);
      vst1q_f32(p + 8 * i + 4, im);
    };

    Base::for_.Run(MinSize(state.num_qubits()) / 8, f2,
                   mr.mask, mr.bits, renorm, zero, state.get());
  }

  std::vector<double> PartialNorms(const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* p) -> double {
      float32x4_t re = vld1q_f32(p + 8 * i);
      float32x4_t im = vld1q_f32(p + 8 * i + 4);
      float32x4_t s1 = vmulq_f32(re, re);
      s1 = vfmaq_f32(s1, im, im);
      return detail::HorizontalSumNEON(s1);
    };

    using Op = std::plus<double>;
    return Base::for_.RunReduceP(
        MinSize(state.num_qubits()) / 8, f, Op(), state.get());
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    double csum = 0;

    uint64_t k0 = Base::for_.GetIndex0(MinSize(state.num_qubits()) / 8, m);
    uint64_t k1 = Base::for_.GetIndex1(MinSize(state.num_qubits()) / 8, m);

    const fp_type* p = state.get();

    for (uint64_t k = k0; k < k1; ++k) {
      for (uint64_t j = 0; j < 4; ++j) {
        auto re = p[8 * k + j];
        auto im = p[8 * k + 4 + j];
        csum += re * re + im * im;
        if (r < csum) {
          return (4 * k + j) & mask;
        }
      }
    }

    return (4 * k1 - 1) & mask;
  }
};

}  // namespace qsim

#endif  // STATESPACE_NEON_H_
