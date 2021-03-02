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

#ifndef UNITARY_CALCULATOR_AVX_H_
#define UNITARY_CALCULATOR_AVX_H_

#include <immintrin.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#include "bits.h"
#include "unitaryspace_avx.h"

namespace qsim {
namespace unitary {

/**
 * Quantum circuit unitary calculator with AVX vectorization.
 */
template <typename For>
class UnitaryCalculatorAVX final {
 public:
  using UnitarySpace = UnitarySpaceAVX<For>;
  using Unitary = typename UnitarySpace::Unitary;
  using fp_type = typename UnitarySpace::fp_type;

  using StateSpace = UnitarySpace;
  using State = Unitary;

  template <typename... ForArgs>
  explicit UnitaryCalculatorAVX(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using AVX instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, Unitary& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 2) {
        ApplyGate1H(qs, matrix, state);
      } else {
        ApplyGate1L(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 2) {
        ApplyGate2HH(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGate2HL(qs, matrix, state);
      } else {
        ApplyGate2LL(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 2) {
        ApplyGate3HHH(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGate3HHL(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGate3HLL(qs, matrix, state);
      } else {
        ApplyGate3LLL(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 2) {
        ApplyGate4HHHH(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGate4HHHL(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGate4HHLL(qs, matrix, state);
      } else {
        ApplyGate4HLLL(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 2) {
        ApplyGate5HHHHH(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGate5HHHHL(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGate5HHHLL(qs, matrix, state);
      } else {
        ApplyGate5HHLLL(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 2) {
        ApplyGate6HHHHHH(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGate6HHHHHL(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGate6HHHHLL(qs, matrix, state);
      } else {
        ApplyGate6HHHLLL(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using AVX instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, Unitary& state) const {
    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate1H_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate1H_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGate1L_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate1L_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 2:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate2HH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2HH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate2HL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2HL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGate2LL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2LL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 3:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate3HHH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HHH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate3HHL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HHL_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[2] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate3HLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HLL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGate3LLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3LLL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 4:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate4HHHH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHHH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate4HHHL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHHL_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[2] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGate4HHLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHLL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGate4HLLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HLLL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 8;
  }

 private:
  void ApplyGate1H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate1L(const std::vector<unsigned>& qs,
                   const fp_type* matrix, Unitary& state) const {
    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(4);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      auto p0 = rstate + row_size * r + 16 * ii;

      for (unsigned l = 0; l < 1; ++l) {
        rs[2 * l] = _mm256_load_ps(p0);
        is[2 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, idx, size, raw_size, rstate);
  }

  void ApplyGate2HH(const std::vector<unsigned>& qs,
                    const fp_type* matrix, Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate2HL(const std::vector<unsigned>& qs,
                    const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate2LL(const std::vector<unsigned>& qs,
                    const fp_type* matrix, Unitary& state) const {
    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(5);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      auto p0 = rstate + row_size * r + 16 * ii;

      for (unsigned l = 0; l < 1; ++l) {
        rs[4 * l] = _mm256_load_ps(p0);
        is[4 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, idx, size, raw_size, rstate);
  }

  void ApplyGate3HHH(const std::vector<unsigned>& qs,
                     const fp_type* matrix, Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate3HHL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate3HLL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(7);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate3LLL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, Unitary& state) const {
    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      auto p0 = rstate + row_size * r + 16 * ii;

      for (unsigned l = 0; l < 1; ++l) {
        rs[8 * l] = _mm256_load_ps(p0);
        is[8 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, idx, size, raw_size, rstate);
  }

  void ApplyGate4HHHH(const std::vector<unsigned>& qs,
                      const fp_type* matrix, Unitary& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate4HHHL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(10);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate4HHLL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(9);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate4HLLL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[8 * l] = _mm256_load_ps(p0 + xss[l]);
        is[8 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate5HHHHH(const std::vector<unsigned>& qs,
                       const fp_type* matrix, Unitary& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[32], is[32];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]) | (256 * ii & ms[5]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 32; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 32; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate5HHHHL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, Unitary& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(12);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 32 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[32], is[32];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 16; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate5HHHLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(11);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (128 * i + 32 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[32], is[32];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate5HHLLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(10);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (256 * i + 32 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[32], is[32];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[8 * l] = _mm256_load_ps(p0 + xss[l]);
        is[8 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate6HHHHHH(const std::vector<unsigned>& qs,
                        const fp_type* matrix, Unitary& state) const {
    uint64_t xs[6];
    uint64_t ms[7];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 6; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[6] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[5] - 1);

    uint64_t xss[64];
    for (unsigned i = 0; i < 64; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 6; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[64], is[64];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]) | (256 * ii & ms[5])
          | (512 * ii & ms[6]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 64; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 64; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, rstate);
  }

  void ApplyGate6HHHHHL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, Unitary& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(14);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 32; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (128 * i + 64 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[64], is[64];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]) | (256 * ii & ms[5]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 32; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 32; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate6HHHHLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, Unitary& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(13);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (256 * i + 64 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[64], is[64];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]) | (128 * ii & ms[4]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 16; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyGate6HHHLLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(12);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (512 * i + 64 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[64], is[64];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = (8 * ii & ms[0]) | (16 * ii & ms[1]) | (32 * ii & ms[2])
          | (64 * ii & ms[3]);

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[8 * l] = _mm256_load_ps(p0 + xss[l]);
        is[8 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate1H_H(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate1H_L(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];

    auto s = UnitarySpace::Create(5);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (2 * i + 2 * k + m);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 2 == (p[j] / 2) % 2 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate1L_H(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               Unitary& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(4);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[2 * l] = _mm256_load_ps(p0);
        is[2 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate1L_L(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               Unitary& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(4);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 2 == (p[j] / 2) % 2 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[2], is[2];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[2 * l] = _mm256_load_ps(p0);
        is[2 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate2HH_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate2HH_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];

    auto s = UnitarySpace::Create(7);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (4 * i + 4 * k + m);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate2HL_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate2HL_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate2LL_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(5);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[4 * l] = _mm256_load_ps(p0);
        is[4 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate2LL_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                Unitary& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(5);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[4], is[4];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[4 * l] = _mm256_load_ps(p0);
        is[4 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3HHH_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate3HHH_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];

    auto s = UnitarySpace::Create(9);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (8 * i + 8 * k + m);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate3HHL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3HHL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3HLL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(7);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3HLL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(7);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3LLL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[8 * l] = _mm256_load_ps(p0);
        is[8 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate3LLL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 Unitary& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(6);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[8], is[8];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 1; ++l) {
        rs[8 * l] = _mm256_load_ps(p0);
        is[8 * l] = _mm256_load_ps(p0 + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 1; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0, rn);
        _mm256_store_ps(p0 + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHHH_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 ru, iu, rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[n], ru, rn);
          in = _mm256_fmadd_ps(rs[n], iu, in);
          rn = _mm256_fnmadd_ps(is[n], iu, rn);
          in = _mm256_fmadd_ps(is[n], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHHH_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];

    auto s = UnitarySpace::Create(11);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (16 * i + 16 * k + m);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = _mm256_load_ps(p0 + xss[l]);
        is[l] = _mm256_load_ps(p0 + xss[l] + 8);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHHL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(10);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHHL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[1];

    auto s = UnitarySpace::Create(10);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 2) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 8; ++l) {
        rs[2 * l] = _mm256_load_ps(p0 + xss[l]);
        is[2 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 2; ++j) {
          rs[2 * l + j] = _mm256_permutevar8x32_ps(rs[2 * l], idx[j - 1]);
          is[2 * l + j] = _mm256_permutevar8x32_ps(is[2 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHLL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(9);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HHLL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[3];

    auto s = UnitarySpace::Create(9);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 4) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[4 * l] = _mm256_load_ps(p0 + xss[l]);
        is[4 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 4; ++j) {
          rs[4 * l + j] = _mm256_permutevar8x32_ps(rs[4 * l], idx[j - 1]);
          is[4 * l + j] = _mm256_permutevar8x32_ps(is[4 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HLLL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = matrix[p[j] + 1];
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[8 * l] = _mm256_load_ps(p0 + xss[l]);
        is[8 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  void ApplyControlledGate4HLLL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 3, emaskl);

    for (auto q : qs) {
      if (q > 2) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 7;

    unsigned p[8];
    __m256i idx[7];

    auto s = UnitarySpace::Create(8);
    __m256* w = (__m256*) s.get();
    fp_type* wf = (fp_type*) w;

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd(j, i + 1, qmask, 8) | (j & (-1 ^ qmask));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned k = bits::CompressBits(j, 3, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 8; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[8 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 8; ++j) {
          wf[8 * l + j + 8] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      __m256 rn, in;
      __m256 rs[16], is[16];

      uint64_t ii = i % size;
      uint64_t r = i / size;
      uint64_t c = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[8 * l] = _mm256_load_ps(p0 + xss[l]);
        is[8 * l] = _mm256_load_ps(p0 + xss[l] + 8);

        for (unsigned j = 1; j < 8; ++j) {
          rs[8 * l + j] = _mm256_permutevar8x32_ps(rs[8 * l], idx[j - 1]);
          is[8 * l + j] = _mm256_permutevar8x32_ps(is[8 * l], idx[j - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn = _mm256_fmadd_ps(rs[n], w[j], rn);
          in = _mm256_fmadd_ps(rs[n], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[n], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[n], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[l], rn);
        _mm256_store_ps(p0 + xss[l] + 8, in);
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss,
             state.num_qubits(), cmaskh, emaskh, idx, size, raw_size, rstate);
  }

  static unsigned MaskedAdd(
      unsigned a, unsigned b, unsigned mask, unsigned lsize) {
    unsigned c = bits::CompressBits(a, 3, mask);
    return bits::ExpandBits((c + b) % lsize, 3, mask);
  }

  For for_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_AVX_H_
