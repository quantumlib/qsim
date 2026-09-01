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

#ifndef SIMULATOR_NEON_H_
#define SIMULATOR_NEON_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "statespace_neon.h"

namespace qsim {

/**
 * Quantum circuit simulator with NEON vectorization.
 */
template <typename For>
class SimulatorNEON final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceNEON<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorNEON(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using NEON instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
      case 0:
        ApplyGateH<0>(qs, matrix, state);
        return;
      case 1:
        if (qs[0] > 1) {
          ApplyGateH<1>(qs, matrix, state);
        } else {
          ApplyGateL<0, 1>(qs, matrix, state);
        }
        return;
      case 2:
        if (qs[0] > 1) {
          ApplyGateH<2>(qs, matrix, state);
        } else if (qs[1] > 1) {
          ApplyGateL<1, 1>(qs, matrix, state);
        } else {
          ApplyGateL<0, 2>(qs, matrix, state);
        }
        return;
      case 3:
        if (qs[0] > 1) {
          ApplyGateH<3>(qs, matrix, state);
        } else if (qs[1] > 1) {
          ApplyGateL<2, 1>(qs, matrix, state);
        } else {
          ApplyGateL<1, 2>(qs, matrix, state);
        }
        return;
      case 4:
        if (qs[0] > 1) {
          ApplyGateH<4>(qs, matrix, state);
        } else if (qs[1] > 1) {
          ApplyGateL<3, 1>(qs, matrix, state);
        } else {
          ApplyGateL<2, 2>(qs, matrix, state);
        }
        return;
      case 5:
        if (qs[0] > 1) {
          ApplyGateH<5>(qs, matrix, state);
        } else if (qs[1] > 1) {
          ApplyGateL<4, 1>(qs, matrix, state);
        } else {
          ApplyGateL<3, 2>(qs, matrix, state);
        }
        return;
      case 6:
        if (qs[0] > 1) {
          ApplyGateH<6>(qs, matrix, state);
        } else if (qs[1] > 1) {
          ApplyGateL<5, 1>(qs, matrix, state);
        } else {
          ApplyGateL<4, 2>(qs, matrix, state);
        }
        return;
      default:
        break;
    }
  }

  /**
   * Applies a controlled gate using NEON instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    // Assume cqs[0] < cqs[1] < cqs[2] < ... .

    if (cqs.empty()) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
      case 0:
        if (cqs[0] > 1) {
          ApplyControlledGateHH<0>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<0>(qs, cqs, cvals, matrix, state);
        }
        return;
      case 1:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<1>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<0, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<0, 1, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      case 2:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<2>(qs, cqs, cvals, matrix, state);
          }
        } else if (qs[1] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateL<1, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<1, 1, false>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<0, 2, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<0, 2, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      case 3:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<3>(qs, cqs, cvals, matrix, state);
          }
        } else if (qs[1] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateL<2, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<2, 1, false>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<1, 2, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<1, 2, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      case 4:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<4>(qs, cqs, cvals, matrix, state);
          }
        } else if (qs[1] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateL<3, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<3, 1, false>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<2, 2, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<2, 2, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      case 5:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<5>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<5>(qs, cqs, cvals, matrix, state);
          }
        } else if (qs[1] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateL<4, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<4, 1, false>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<3, 2, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<3, 2, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      case 6:
        if (qs[0] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateHH<6>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateHL<6>(qs, cqs, cvals, matrix, state);
          }
        } else if (qs[1] > 1) {
          if (cqs[0] > 1) {
            ApplyControlledGateL<5, 1, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<5, 1, false>(qs, cqs, cvals, matrix, state);
          }
        } else {
          if (cqs[0] > 1) {
            ApplyControlledGateL<4, 2, true>(qs, cqs, cvals, matrix, state);
          } else {
            ApplyControlledGateL<4, 2, false>(qs, cqs, cvals, matrix, state);
          }
        }
        return;
      default:
        break;
    }
  }

  std::complex<double> ExpectationValue(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    switch (qs.size()) {
      case 1:
        if (qs[0] > 1) {
          return ExpectationValueH<1>(qs, matrix, state);
        } else {
          return ExpectationValueL<0, 1>(qs, matrix, state);
        }
      case 2:
        if (qs[0] > 1) {
          return ExpectationValueH<2>(qs, matrix, state);
        } else if (qs[1] > 1) {
          return ExpectationValueL<1, 1>(qs, matrix, state);
        } else {
          return ExpectationValueL<0, 2>(qs, matrix, state);
        }
      case 3:
        if (qs[0] > 1) {
          return ExpectationValueH<3>(qs, matrix, state);
        } else if (qs[1] > 1) {
          return ExpectationValueL<2, 1>(qs, matrix, state);
        } else {
          return ExpectationValueL<1, 2>(qs, matrix, state);
        }
      case 4:
        if (qs[0] > 1) {
          return ExpectationValueH<4>(qs, matrix, state);
        } else if (qs[1] > 1) {
          return ExpectationValueL<3, 1>(qs, matrix, state);
        } else {
          return ExpectationValueL<2, 2>(qs, matrix, state);
        }
      case 5:
        if (qs[0] > 1) {
          return ExpectationValueH<5>(qs, matrix, state);
        } else if (qs[1] > 1) {
          return ExpectationValueL<4, 1>(qs, matrix, state);
        } else {
          return ExpectationValueL<3, 2>(qs, matrix, state);
        }
      case 6:
        if (qs[0] > 1) {
          return ExpectationValueH<6>(qs, matrix, state);
        } else if (qs[1] > 1) {
          return ExpectationValueL<5, 1>(qs, matrix, state);
        } else {
          return ExpectationValueL<4, 2>(qs, matrix, state);
        }
      default:
        break;
    }
    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 4;
  }

 private:
  template <unsigned H>
  void ApplyGateH(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      float32x4_t rs[hsize];
      float32x4_t is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = vld1q_f32(p0 + xss[k]);
        is[k] = vld1q_f32(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t ru = vdupq_n_f32(v[j]);
        float32x4_t iu = vdupq_n_f32(v[j + 1]);
        float32x4_t rn = vmulq_f32(rs[0], ru);
        float32x4_t in = vmulq_f32(rs[0], iu);
        rn = vfmsq_f32(rn, is[0], iu);
        in = vfmaq_f32(in, is[0], ru);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = vdupq_n_f32(v[j]);
          iu = vdupq_n_f32(v[j + 1]);
          rn = vfmaq_f32(rn, rs[l], ru);
          in = vfmaq_f32(in, rs[l], iu);
          rn = vfmsq_f32(rn, is[l], iu);
          in = vfmaq_f32(in, is[l], ru);

          j += 2;
        }

        vst1q_f32(p0 + xss[k], rn);
        vst1q_f32(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  void ApplyGateL(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* w,
                const uint64_t* ms, const uint64_t* xss, unsigned q0,
                fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      float32x4_t rs[gsize];
      float32x4_t is[gsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = vld1q_f32(p0 + xss[k]);
        is[k2] = vld1q_f32(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] =
              q0 == 0 ? vrev64q_f32(rs[k2]) : vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 1] =
              q0 == 0 ? vrev64q_f32(is[k2]) : vextq_f32(is[k2], is[k2], 2);
        } else if (L == 2) {
          rs[k2 + 1] = vextq_f32(rs[k2], rs[k2], 1);
          is[k2 + 1] = vextq_f32(is[k2], is[k2], 1);
          rs[k2 + 2] = vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 2] = vextq_f32(is[k2], is[k2], 2);
          rs[k2 + 3] = vextq_f32(rs[k2], rs[k2], 3);
          is[k2 + 3] = vextq_f32(is[k2], is[k2], 3);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t wre = vld1q_f32(w + 4 * j);
        float32x4_t wim = vld1q_f32(w + 4 * (j + 1));
        float32x4_t rn = vmulq_f32(rs[0], wre);
        float32x4_t in = vmulq_f32(rs[0], wim);
        rn = vfmsq_f32(rn, is[0], wim);
        in = vfmaq_f32(in, is[0], wre);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          wre = vld1q_f32(w + 4 * j);
          wim = vld1q_f32(w + 4 * (j + 1));
          rn = vfmaq_f32(rn, rs[l], wre);
          in = vfmaq_f32(in, rs[l], wim);
          rn = vfmsq_f32(rn, is[l], wim);
          in = vfmaq_f32(in, is[l], wre);

          j += 2;
        }

        vst1q_f32(p0 + xss[k], rn);
        vst1q_f32(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    alignas(16) fp_type w[1 << (3 + 2 * H + L)];

    auto m = GetMasks11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillMatrix<H, L, 2>(m.qmaskl, matrix, w);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, w, ms, xss, qs[0], state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHH(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      float32x4_t rs[hsize];
      float32x4_t is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = vld1q_f32(p0 + xss[k]);
        is[k] = vld1q_f32(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t ru = vdupq_n_f32(v[j]);
        float32x4_t iu = vdupq_n_f32(v[j + 1]);
        float32x4_t rn = vmulq_f32(rs[0], ru);
        float32x4_t in = vmulq_f32(rs[0], iu);
        rn = vfmsq_f32(rn, is[0], iu);
        in = vfmaq_f32(in, is[0], ru);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = vdupq_n_f32(v[j]);
          iu = vdupq_n_f32(v[j + 1]);
          rn = vfmaq_f32(rn, rs[l], ru);
          in = vfmaq_f32(in, rs[l], iu);
          rn = vfmsq_f32(rn, is[l], iu);
          in = vfmaq_f32(in, is[l], ru);

          j += 2;
        }

        vst1q_f32(p0 + xss[k], rn);
        vst1q_f32(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHL(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      float32x4_t rs[hsize];
      float32x4_t is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = vld1q_f32(p0 + xss[k]);
        is[k] = vld1q_f32(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t wre = vld1q_f32(w + 4 * j);
        float32x4_t wim = vld1q_f32(w + 4 * (j + 1));
        float32x4_t rn = vmulq_f32(rs[0], wre);
        float32x4_t in = vmulq_f32(rs[0], wim);
        rn = vfmsq_f32(rn, is[0], wim);
        in = vfmaq_f32(in, is[0], wre);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          wre = vld1q_f32(w + 4 * j);
          wim = vld1q_f32(w + 4 * (j + 1));
          rn = vfmaq_f32(rn, rs[l], wre);
          in = vfmaq_f32(in, rs[l], wim);
          rn = vfmsq_f32(rn, is[l], wim);
          in = vfmaq_f32(in, is[l], wre);

          j += 2;
        }

        vst1q_f32(p0 + xss[k], rn);
        vst1q_f32(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    alignas(16) fp_type w[1 << (3 + 2 * H)];

    auto m = GetMasks8<2>(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);
    FillControlledMatrixH<H, 2>(m.cvalsl, m.cmaskl, matrix, w);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H, unsigned L, bool CH>
  void ApplyControlledGateL(
      const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
      uint64_t cvals, const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, unsigned q0, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      float32x4_t rs[gsize];
      float32x4_t is[gsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = vld1q_f32(p0 + xss[k]);
        is[k2] = vld1q_f32(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] =
              q0 == 0 ? vrev64q_f32(rs[k2]) : vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 1] =
              q0 == 0 ? vrev64q_f32(is[k2]) : vextq_f32(is[k2], is[k2], 2);
        } else if (L == 2) {
          rs[k2 + 1] = vextq_f32(rs[k2], rs[k2], 1);
          is[k2 + 1] = vextq_f32(is[k2], is[k2], 1);
          rs[k2 + 2] = vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 2] = vextq_f32(is[k2], is[k2], 2);
          rs[k2 + 3] = vextq_f32(rs[k2], rs[k2], 3);
          is[k2 + 3] = vextq_f32(is[k2], is[k2], 3);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t wre = vld1q_f32(w + 4 * j);
        float32x4_t wim = vld1q_f32(w + 4 * (j + 1));
        float32x4_t rn = vmulq_f32(rs[0], wre);
        float32x4_t in = vmulq_f32(rs[0], wim);
        rn = vfmsq_f32(rn, is[0], wim);
        in = vfmaq_f32(in, is[0], wre);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          wre = vld1q_f32(w + 4 * j);
          wim = vld1q_f32(w + 4 * (j + 1));
          rn = vfmaq_f32(rn, rs[l], wre);
          in = vfmaq_f32(in, rs[l], wim);
          rn = vfmsq_f32(rn, is[l], wim);
          in = vfmaq_f32(in, is[l], wre);

          j += 2;
        }

        vst1q_f32(p0 + xss[k], rn);
        vst1q_f32(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    alignas(16) fp_type w[1 << (3 + 2 * H + L)];

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    if (CH) {
      auto m = GetMasks9<L>(state.num_qubits(), qs, cqs, cvals);
      FillMatrix<H, L, 2>(m.qmaskl, matrix, w);

      for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, qs[0], state.get());
    } else {
      auto m = GetMasks10<L, 2>(state.num_qubits(), qs, cqs, cvals);
      FillControlledMatrixL<H, L, 2>(m.cvalsl, m.cmaskl, m.qmaskl, matrix, w);

      for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, qs[0], state.get());
    }
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) -> std::complex<double> {
      constexpr unsigned hsize = 1 << H;

      float32x4_t rs[hsize];
      float32x4_t is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = vld1q_f32(p0 + xss[k]);
        is[k] = vld1q_f32(p0 + xss[k] + 4);
      }

      double re = 0;
      double im = 0;
      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t ru = vdupq_n_f32(v[j]);
        float32x4_t iu = vdupq_n_f32(v[j + 1]);
        float32x4_t rn = vmulq_f32(rs[0], ru);
        float32x4_t in = vmulq_f32(rs[0], iu);
        rn = vfmsq_f32(rn, is[0], iu);
        in = vfmaq_f32(in, is[0], ru);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = vdupq_n_f32(v[j]);
          iu = vdupq_n_f32(v[j + 1]);
          rn = vfmaq_f32(rn, rs[l], ru);
          in = vfmaq_f32(in, rs[l], iu);
          rn = vfmsq_f32(rn, is[l], iu);
          in = vfmaq_f32(in, is[l], ru);
          j += 2;
        }

        float32x4_t v_re = vmulq_f32(rs[k], rn);
        v_re = vfmaq_f32(v_re, is[k], in);
        float32x4_t v_im = vmulq_f32(rs[k], in);
        v_im = vfmsq_f32(v_im, is[k], rn);

        re += detail::HorizontalSumNEON(v_re);
        im += detail::HorizontalSumNEON(v_im);
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  std::complex<double> ExpectationValueL(
      const std::vector<unsigned>& qs, const fp_type* matrix,
      const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* w,
                const uint64_t* ms, const uint64_t* xss, unsigned q0,
                const fp_type* rstate) -> std::complex<double> {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      float32x4_t rs[gsize];
      float32x4_t is[gsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = vld1q_f32(p0 + xss[k]);
        is[k2] = vld1q_f32(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] =
              q0 == 0 ? vrev64q_f32(rs[k2]) : vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 1] =
              q0 == 0 ? vrev64q_f32(is[k2]) : vextq_f32(is[k2], is[k2], 2);
        } else if (L == 2) {
          rs[k2 + 1] = vextq_f32(rs[k2], rs[k2], 1);
          is[k2 + 1] = vextq_f32(is[k2], is[k2], 1);
          rs[k2 + 2] = vextq_f32(rs[k2], rs[k2], 2);
          is[k2 + 2] = vextq_f32(is[k2], is[k2], 2);
          rs[k2 + 3] = vextq_f32(rs[k2], rs[k2], 3);
          is[k2 + 3] = vextq_f32(is[k2], is[k2], 3);
        }
      }

      double re = 0;
      double im = 0;
      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        float32x4_t wre = vld1q_f32(w + 4 * j);
        float32x4_t wim = vld1q_f32(w + 4 * (j + 1));
        float32x4_t rn = vmulq_f32(rs[0], wre);
        float32x4_t in = vmulq_f32(rs[0], wim);
        rn = vfmsq_f32(rn, is[0], wim);
        in = vfmaq_f32(in, is[0], wre);
        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          wre = vld1q_f32(w + 4 * j);
          wim = vld1q_f32(w + 4 * (j + 1));
          rn = vfmaq_f32(rn, rs[l], wre);
          in = vfmaq_f32(in, rs[l], wim);
          rn = vfmsq_f32(rn, is[l], wim);
          in = vfmaq_f32(in, is[l], wre);
          j += 2;
        }

        unsigned m = lsize * k;
        float32x4_t v_re = vmulq_f32(rs[m], rn);
        v_re = vfmaq_f32(v_re, is[m], in);
        float32x4_t v_im = vmulq_f32(rs[m], in);
        v_im = vfmsq_f32(v_im, is[m], rn);

        re += detail::HorizontalSumNEON(v_re);
        im += detail::HorizontalSumNEON(v_im);
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    alignas(16) fp_type w[1 << (3 + 2 * H + L)];

    auto m = GetMasks11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillMatrix<H, L, 2>(m.qmaskl, matrix, w);

    const unsigned k = 2 + H;
    const unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    const uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), w, ms, xss, qs[0], state.get());
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_NEON_H_
