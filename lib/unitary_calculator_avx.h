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

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "unitaryspace_avx.h"

namespace qsim {
namespace unitary {

/**
 * Quantum circuit unitary calculator with AVX vectorization.
 */
template <typename For>
class UnitaryCalculatorAVX final : public SimulatorBase {
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
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 2) {
        ApplyGateH<1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 1>(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 2) {
        ApplyGateH<2>(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGateL<1, 1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 2>(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 2) {
        ApplyGateH<3>(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGateL<2, 1>(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGateL<1, 2>(qs, matrix, state);
      } else {
        ApplyGateL<0, 3>(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 2) {
        ApplyGateH<4>(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGateL<3, 1>(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGateL<2, 2>(qs, matrix, state);
      } else {
        ApplyGateL<1, 3>(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 2) {
        ApplyGateH<5>(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGateL<4, 1>(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGateL<3, 2>(qs, matrix, state);
      } else {
        ApplyGateL<2, 3>(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 2) {
        ApplyGateH<6>(qs, matrix, state);
      } else if (qs[1] > 2) {
        ApplyGateL<5, 1>(qs, matrix, state);
      } else if (qs[2] > 2) {
        ApplyGateL<4, 2>(qs, matrix, state);
      } else {
        ApplyGateL<3, 3>(qs, matrix, state);
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
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<1>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGateL<0, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 2:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<2>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateL<1, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGateL<0, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 3:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<3>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateL<2, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[2] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateL<1, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGateL<0, 3, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 3, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 4:
      if (qs[0] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<4>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateL<3, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<3, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[2] > 2) {
        if (cqs[0] > 2) {
          ApplyControlledGateL<2, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 2) {
          ApplyControlledGateL<1, 3, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 3, 0>(qs, cqs, cvals, matrix, state);
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

#ifdef __BMI2__

  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                uint64_t imaskh, uint64_t qmaskh, uint64_t size,
                uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 ru, iu, rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + _pdep_u64(r, imaskh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm256_load_ps(p0 + p);
        is[k] = _mm256_load_ps(p0 + p + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[l], ru, rn);
          in = _mm256_fmadd_ps(rs[l], iu, in);
          rn = _mm256_fnmadd_ps(is[l], iu, rn);
          in = _mm256_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm256_store_ps(p0 + p, rn);
        _mm256_store_ps(p0 + p + 8, in);
      }
    };

    auto m = GetMasks1<H, 3>(qs);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             matrix, m.imaskh, m.qmaskh, size, raw_size, state.get());
  }

  template <unsigned H, unsigned L>
  void ApplyGateL(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                uint64_t imaskh, uint64_t qmaskh, const __m256i* idx,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m256 rn, in;
      __m256 rs[gsize], is[gsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + _pdep_u64(r, imaskh);

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k2] = _mm256_load_ps(p0 + p);
        is[k2] = _mm256_load_ps(p0 + p + 8);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm256_permutevar8x32_ps(rs[k2], idx[l - 1]);
          is[k2 + l] = _mm256_permutevar8x32_ps(is[k2], idx[l - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm256_store_ps(p0 + p, rn);
        _mm256_store_ps(p0 + p + 8, in);
      }
    };

    __m256i idx[1 << L];
    __m256 w[1 << (1 + 2 * H + L)];

    auto m = GetMasks2<H, L, 3>(qs);
    FillPermutationIndices<L>(m.qmaskl, idx);
    FillMatrix<H, L, 3>(m.qmaskl, matrix, (fp_type*) w);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             w, m.imaskh, m.qmaskh, idx, size, raw_size, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                uint64_t imaskh, uint64_t qmaskh, uint64_t cvalsh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 ru, iu, rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm256_load_ps(p0 + p);
        is[k] = _mm256_load_ps(p0 + p + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[l], ru, rn);
          in = _mm256_fmadd_ps(rs[l], iu, in);
          rn = _mm256_fnmadd_ps(is[l], iu, rn);
          in = _mm256_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm256_store_ps(p0 + p, rn);
        _mm256_store_ps(p0 + p + 8, in);
      }
    };

    auto m = GetMasks3<H, 3>(state.num_qubits(), qs, cqs, cvals);

    unsigned k = 3 + H + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             matrix, m.imaskh, m.qmaskh, m.cvalsh, size, raw_size, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHL(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                uint64_t imaskh, uint64_t qmaskh, uint64_t cvalsh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm256_load_ps(p0 + p);
        is[k] = _mm256_load_ps(p0 + p + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm256_store_ps(p0 + p, rn);
        _mm256_store_ps(p0 + p + 8, in);
      }
    };

    __m256 w[1 << (1 + 2 * H)];

    auto m = GetMasks4<H, 3>(state.num_qubits(), qs, cqs, cvals);
    FillControlledMatrixH<H, 3>(m.cvalsl, m.cmaskl, matrix, (fp_type*) w);

    unsigned k = 3 + H + cqs.size() - m.cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             w, m.imaskh, m.qmaskh, m.cvalsh, size, raw_size, state.get());
  }

  template <unsigned H, unsigned L, bool CH>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                uint64_t imaskh, uint64_t qmaskh, uint64_t cvalsh,
                const __m256i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m256 rn, in;
      __m256 rs[gsize], is[gsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k2] = _mm256_load_ps(p0 + p);
        is[k2] = _mm256_load_ps(p0 + p + 8);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm256_permutevar8x32_ps(rs[k2], idx[l - 1]);
          is[k2 + l] = _mm256_permutevar8x32_ps(is[k2], idx[l - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm256_store_ps(p0 + p, rn);
        _mm256_store_ps(p0 + p + 8, in);
      }
    };

    __m256i idx[1 << L];
    __m256 w[1 << (1 + 2 * H + L)];

    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    if (CH) {
      auto m = GetMasks5<H, L, 3>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillMatrix<H, L, 3>(m.qmaskl, matrix, (fp_type*) w);

      unsigned k = 3 + H + cqs.size();
      unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size * size2, f, w, m.imaskh, m.qmaskh,
               m.cvalsh, idx, size, raw_size, state.get());
    } else {
      auto m = GetMasks6<H, L, 3>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillControlledMatrixL<H, L, 3>(
          m.cvalsl, m.cmaskl, m.qmaskl, matrix, (fp_type*) w);

      unsigned k = 3 + H + cqs.size() - m.cl;
      unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size * size2, f, w, m.imaskh, m.qmaskh,
               m.cvalsh, idx, size, raw_size, state.get());
    }
  }

#else  // __BMI2__

  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t size,
                uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 ru, iu, rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = 8 * (i % size);
      uint64_t s = i / size;

      uint64_t t = r & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        r *= 2;
        t |= r & ms[j];
      }

      auto p0 = rstate + row_size * s + 2 * t;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm256_load_ps(p0 + xss[k]);
        is[k] = _mm256_load_ps(p0 + xss[k] + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[l], ru, rn);
          in = _mm256_fmadd_ps(rs[l], iu, in);
          rn = _mm256_fnmadd_ps(is[l], iu, rn);
          in = _mm256_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[k], rn);
        _mm256_store_ps(p0 + xss[k] + 8, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, matrix, ms, xss, size, raw_size, state.get());
  }

  template <unsigned H, unsigned L>
  void ApplyGateL(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss, const __m256i* idx,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m256 rn, in;
      __m256 rs[gsize], is[gsize];

      uint64_t r = 8 * (i % size);
      uint64_t s = i / size;

      uint64_t t = r & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        r *= 2;
        t |= r & ms[j];
      }

      auto p0 = rstate + row_size * s + 2 * t;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        rs[k2] = _mm256_load_ps(p0 + xss[k]);
        is[k2] = _mm256_load_ps(p0 + xss[k] + 8);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm256_permutevar8x32_ps(rs[k2], idx[l - 1]);
          is[k2 + l] = _mm256_permutevar8x32_ps(is[k2], idx[l - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[k], rn);
        _mm256_store_ps(p0 + xss[k] + 8, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m256i idx[1 << L];
    __m256 w[1 << (1 + 2 * H + L)];

    auto m = GetMasks11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillPermutationIndices<L>(m.qmaskl, idx);
    FillMatrix<H, L, 3>(m.qmaskl, matrix, (fp_type*) w);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f, w, ms, xss, idx, size, raw_size, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 ru, iu, rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = 8 * (i % size);
      uint64_t s = i / size;

      uint64_t t = r & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        r *= 2;
        t |= r & ms[j];
      }

      if ((t & cmaskh) != cvalsh) return;

      auto p0 = rstate + row_size * s + 2 * t;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm256_load_ps(p0 + xss[k]);
        is[k] = _mm256_load_ps(p0 + xss[k] + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm256_set1_ps(v[j]);
        iu = _mm256_set1_ps(v[j + 1]);
        rn = _mm256_mul_ps(rs[0], ru);
        in = _mm256_mul_ps(rs[0], iu);
        rn = _mm256_fnmadd_ps(is[0], iu, rn);
        in = _mm256_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm256_set1_ps(v[j]);
          iu = _mm256_set1_ps(v[j + 1]);
          rn = _mm256_fmadd_ps(rs[l], ru, rn);
          in = _mm256_fmadd_ps(rs[l], iu, in);
          rn = _mm256_fnmadd_ps(is[l], iu, rn);
          in = _mm256_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[k], rn);
        _mm256_store_ps(p0 + xss[k] + 8, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             matrix, ms, xss, m.cvalsh, m.cmaskh, size, raw_size, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateHL(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m256 rn, in;
      __m256 rs[hsize], is[hsize];

      uint64_t r = 8 * (i % size);
      uint64_t s = i / size;

      uint64_t t = r & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        r *= 2;
        t |= r & ms[j];
      }

      if ((t & cmaskh) != cvalsh) return;

      auto p0 = rstate + row_size * s + 2 * t;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = _mm256_load_ps(p0 + xss[k]);
        is[k] = _mm256_load_ps(p0 + xss[k] + 8);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[k], rn);
        _mm256_store_ps(p0 + xss[k] + 8, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m256 w[1 << (1 + 2 * H)];

    auto m = GetMasks8<3>(state.num_qubits(), qs, cqs, cvals);
    FillIndices<H>(state.num_qubits(), qs, ms, xss);
    FillControlledMatrixH<H, 3>(m.cvalsl, m.cmaskl, matrix, (fp_type*) w);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    for_.Run(size * size2, f,
             w, ms, xss, m.cvalsh, m.cmaskh, size, raw_size, state.get());
  }

  template <unsigned H, unsigned L, bool CH>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m256* w,
                const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
                uint64_t cmaskh, const __m256i* idx, uint64_t size,
                uint64_t row_size, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m256 rn, in;
      __m256 rs[gsize], is[gsize];

      uint64_t r = 8 * (i % size);
      uint64_t s = i / size;

      uint64_t t = r & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        r *= 2;
        t |= r & ms[j];
      }

      if ((t & cmaskh) != cvalsh) return;

      auto p0 = rstate + row_size * s + 2 * t;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = _mm256_load_ps(p0 + xss[k]);
        is[k2] = _mm256_load_ps(p0 + xss[k] + 8);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm256_permutevar8x32_ps(rs[k2], idx[l - 1]);
          is[k2 + l] = _mm256_permutevar8x32_ps(is[k2], idx[l - 1]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm256_mul_ps(rs[0], w[j]);
        in = _mm256_mul_ps(rs[0], w[j + 1]);
        rn = _mm256_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm256_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm256_fmadd_ps(rs[l], w[j], rn);
          in = _mm256_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm256_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm256_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        _mm256_store_ps(p0 + xss[k], rn);
        _mm256_store_ps(p0 + xss[k] + 8, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m256i idx[1 << L];
    __m256 w[1 << (1 + 2 * H + L)];

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

    unsigned k = 3 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    if (CH) {
      auto m = GetMasks9<L>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillMatrix<H, L, 3>(m.qmaskl, matrix, (fp_type*) w);

      for_.Run(size * size2, f, w, ms, xss, m.cvalsh,
               m.cmaskh, idx, size, raw_size, state.get());
    } else {
      auto m = GetMasks10<L, 3>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillControlledMatrixL<H, L, 3>(
          m.cvalsl, m.cmaskl, m.qmaskl, matrix, (fp_type*) w);

      for_.Run(size * size2, f, w, ms, xss, m.cvalsh,
               m.cmaskh, idx, size, raw_size, state.get());
    }
  }

#endif  // __BMI2__

  template <unsigned L>
  static void FillPermutationIndices(unsigned qmaskl, __m256i* idx) {
    constexpr unsigned lsize = 1 << L;

    for (unsigned i = 0; i < lsize - 1; ++i) {
      unsigned p[8];

      for (unsigned j = 0; j < 8; ++j) {
        p[j] = MaskedAdd<3>(j, i + 1, qmaskl, lsize) | (j & (-1 ^ qmaskl));
      }

      idx[i] = _mm256_set_epi32(p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
    }
  }

  For for_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_AVX_H_
