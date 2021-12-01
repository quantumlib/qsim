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

#ifndef SIMULATOR_SSE_H_
#define SIMULATOR_SSE_H_

#include <smmintrin.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "statespace_sse.h"

namespace qsim {

/**
 * Quantum circuit simulator with SSE vectorization.
 */
template <typename For>
class SimulatorSSE final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceSSE<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorSSE(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using SSE instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 1) {
        ApplyGateH<1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 1>(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 1) {
        ApplyGateH<2>(qs, matrix, state);
      } else if (qs[1] > 1) {
        ApplyGateL<1, 1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 2>(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 1) {
        ApplyGateH<3>(qs, matrix, state);
      } else if (qs[1] > 1) {
        ApplyGateL<2, 1>(qs, matrix, state);
      } else {
        ApplyGateL<1, 2>(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 1) {
        ApplyGateH<4>(qs, matrix, state);
      } else if (qs[1] > 1) {
        ApplyGateL<3, 1>(qs, matrix, state);
      } else {
        ApplyGateL<2, 2>(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 1) {
        ApplyGateH<5>(qs, matrix, state);
      } else if (qs[1] > 1) {
        ApplyGateL<4, 1>(qs, matrix, state);
      } else {
        ApplyGateL<3, 2>(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 1) {
        ApplyGateH<6>(qs, matrix, state);
      } else if (qs[1] > 1) {
        ApplyGateL<5, 1>(qs, matrix, state);
      } else {
        ApplyGateL<4, 2>(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using SSE instructions.
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

    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      if (qs[0] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<1>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 1) {
          ApplyControlledGateL<0, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 2:
      if (qs[0] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<2>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateL<1, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 1) {
          ApplyControlledGateL<0, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 3:
      if (qs[0] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<3>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateL<2, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 1) {
          ApplyControlledGateL<1, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 4:
      if (qs[0] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<4>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 1) {
        if (cqs[0] > 1) {
          ApplyControlledGateL<3, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<3, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 1) {
          ApplyControlledGateL<2, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Computes the expectation value of an operator using SSE instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 1) {
        return ExpectationValueH<1>(qs, matrix, state);
      } else {
        return ExpectationValueL<0, 1>(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 1) {
        return ExpectationValueH<2>(qs, matrix, state);
      } else if (qs[1] > 1) {
        return ExpectationValueL<1, 1>(qs, matrix, state);
      } else {
        return ExpectationValueL<0, 2>(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 1) {
        return ExpectationValueH<3>(qs, matrix, state);
      } else if (qs[1] > 1) {
        return ExpectationValueL<2, 1>(qs, matrix, state);
      } else {
        return ExpectationValueL<1, 2>(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 1) {
        return ExpectationValueH<4>(qs, matrix, state);
      } else if (qs[1] > 1) {
        return ExpectationValueL<3, 1>(qs, matrix, state);
      } else {
        return ExpectationValueL<2, 2>(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 1) {
        return ExpectationValueH<5>(qs, matrix, state);
      } else if (qs[1] > 1) {
        return ExpectationValueL<4, 1>(qs, matrix, state);
      } else {
        return ExpectationValueL<3, 2>(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 1) {
        return ExpectationValueH<6>(qs, matrix, state);
      } else if (qs[1] > 1) {
        return ExpectationValueL<5, 1>(qs, matrix, state);
      } else {
        return ExpectationValueL<4, 2>(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
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
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m128 ru, iu, rn, in;
      __m128 rs[hsize], is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm_load_ps(p0 + xss[k]);
        is[k] = _mm_load_ps(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm_set1_ps(v[j]);
        iu = _mm_set1_ps(v[j + 1]);
        rn = _mm_mul_ps(rs[0], ru);
        in = _mm_mul_ps(rs[0], iu);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], iu));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], ru));

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm_set1_ps(v[j]);
          iu = _mm_set1_ps(v[j + 1]);
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], ru));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], iu));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], iu));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], ru));

          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  void ApplyGateL(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned q0, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m128 rn, in;
      __m128 rs[gsize], is[gsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = _mm_load_ps(p0 + xss[k]);
        is[k2] = _mm_load_ps(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(rs[k2], rs[k2], 177)
                               : _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(is[k2], is[k2], 177)
                               : _mm_shuffle_ps(is[k2], is[k2], 78);
        } else if (L == 2) {
          rs[k2 + 1] = _mm_shuffle_ps(rs[k2], rs[k2], 57);
          is[k2 + 1] = _mm_shuffle_ps(is[k2], is[k2], 57);
          rs[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 2] = _mm_shuffle_ps(is[k2], is[k2], 78);
          rs[k2 + 3] = _mm_shuffle_ps(rs[k2], rs[k2], 147);
          is[k2 + 3] = _mm_shuffle_ps(is[k2], is[k2], 147);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm_mul_ps(rs[0], w[j]);
        in = _mm_mul_ps(rs[0], w[j + 1]);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], w[j]));

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m128 w[1 << (1 + 2 * H + L)];

    auto m = GetMasks11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillMatrix<H, L, 2>(m.qmaskl, matrix, (fp_type*) w);

    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, w, ms, xss, qs[0], state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m128 ru, iu, rn, in;
      __m128 rs[hsize], is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm_load_ps(p0 + xss[k]);
        is[k] = _mm_load_ps(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm_set1_ps(v[j]);
        iu = _mm_set1_ps(v[j + 1]);
        rn = _mm_mul_ps(rs[0], ru);
        in = _mm_mul_ps(rs[0], iu);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], iu));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], ru));

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm_set1_ps(v[j]);
          iu = _mm_set1_ps(v[j + 1]);
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], ru));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], iu));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], iu));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], ru));

          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHL(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m128 rn, in;
      __m128 rs[hsize], is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) != cvalsh) return;

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm_load_ps(p0 + xss[k]);
        is[k] = _mm_load_ps(p0 + xss[k] + 4);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm_mul_ps(rs[0], w[j]);
        in = _mm_mul_ps(rs[0], w[j + 1]);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], w[j]));

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m128 w[1 << (1 + 2 * H)];

    auto m = GetMasks8<2>(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);
    FillControlledMatrixH<H, 2>(m.cvalsl, m.cmaskl, matrix, (fp_type*) w);

    unsigned r = 2 + H;
    unsigned n = state.num_qubits() > r ? state.num_qubits() - r : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H, unsigned L, bool CH>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, unsigned q0, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m128 rn, in;
      __m128 rs[gsize], is[gsize];

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

        rs[k2] = _mm_load_ps(p0 + xss[k]);
        is[k2] = _mm_load_ps(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(rs[k2], rs[k2], 177)
                               : _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(is[k2], is[k2], 177)
                               : _mm_shuffle_ps(is[k2], is[k2], 78);
        } else if (L == 2) {
          rs[k2 + 1] = _mm_shuffle_ps(rs[k2], rs[k2], 57);
          is[k2 + 1] = _mm_shuffle_ps(is[k2], is[k2], 57);
          rs[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 2] = _mm_shuffle_ps(is[k2], is[k2], 78);
          rs[k2 + 3] = _mm_shuffle_ps(rs[k2], rs[k2], 147);
          is[k2 + 3] = _mm_shuffle_ps(is[k2], is[k2], 147);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm_mul_ps(rs[0], w[j]);
        in = _mm_mul_ps(rs[0], w[j + 1]);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], w[j]));

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m128 w[1 << (1 + 2 * H + L)];

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    unsigned r = 2 + H;
    unsigned n = state.num_qubits() > r ? state.num_qubits() - r : 0;
    uint64_t size = uint64_t{1} << n;

    if (CH) {
      auto m = GetMasks9<L>(state.num_qubits(), qs, cqs, cvals);
      FillMatrix<H, L, 2>(m.qmaskl, matrix, (fp_type*) w);

      for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, qs[0], state.get());
    } else {
      auto m = GetMasks10<L, 2>(state.num_qubits(), qs, cqs, cvals);
      FillControlledMatrixL<H, L, 2>(
          m.cvalsl, m.cmaskl, m.qmaskl, matrix, (fp_type*) w);

      for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaskh, qs[0], state.get());
    }
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m128 ru, iu, rn, in;
      __m128 rs[hsize], is[hsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm_load_ps(p0 + xss[k]);
        is[k] = _mm_load_ps(p0 + xss[k] + 4);
      }

      double re = 0;
      double im = 0;
      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm_set1_ps(v[j]);
        iu = _mm_set1_ps(v[j + 1]);
        rn = _mm_mul_ps(rs[0], ru);
        in = _mm_mul_ps(rs[0], iu);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], iu));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], ru));

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm_set1_ps(v[j]);
          iu = _mm_set1_ps(v[j + 1]);
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], ru));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], iu));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], iu));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], ru));

          j += 2;
        }

        __m128 v_re = _mm_add_ps(_mm_mul_ps(rs[k], rn), _mm_mul_ps(is[k], in));
        __m128 v_im = _mm_sub_ps(_mm_mul_ps(rs[k], in), _mm_mul_ps(is[k], rn));

        re += detail::HorizontalSumSSE(v_re);
        im += detail::HorizontalSumSSE(v_im);
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  template <unsigned H, unsigned L>
  std::complex<double> ExpectationValueL(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
                const uint64_t* ms, const uint64_t* xss, unsigned q0,
                const fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m128 rn, in;
      __m128 rs[gsize], is[gsize];

      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = _mm_load_ps(p0 + xss[k]);
        is[k2] = _mm_load_ps(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(rs[k2], rs[k2], 177)
                               : _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(is[k2], is[k2], 177)
                               : _mm_shuffle_ps(is[k2], is[k2], 78);
        } else if (L == 2) {
          rs[k2 + 1] = _mm_shuffle_ps(rs[k2], rs[k2], 57);
          is[k2 + 1] = _mm_shuffle_ps(is[k2], is[k2], 57);
          rs[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 2] = _mm_shuffle_ps(is[k2], is[k2], 78);
          rs[k2 + 3] = _mm_shuffle_ps(rs[k2], rs[k2], 147);
          is[k2 + 3] = _mm_shuffle_ps(is[k2], is[k2], 147);
        }
      }

      double re = 0;
      double im = 0;
      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm_mul_ps(rs[0], w[j]);
        in = _mm_mul_ps(rs[0], w[j + 1]);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], w[j]));

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

          j += 2;
        }

        unsigned m = lsize * k;

        __m128 v_re = _mm_add_ps(_mm_mul_ps(rs[m], rn), _mm_mul_ps(is[m], in));
        __m128 v_im = _mm_sub_ps(_mm_mul_ps(rs[m], in), _mm_mul_ps(is[m], rn));

        re += detail::HorizontalSumSSE(v_re);
        im += detail::HorizontalSumSSE(v_im);
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m128 w[1 << (1 + 2 * H + L)];

    auto m = GetMasks11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillMatrix<H, L, 2>(m.qmaskl, matrix, (fp_type*) w);

    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), w, ms, xss, qs[0], state.get());
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_SSE_H_
