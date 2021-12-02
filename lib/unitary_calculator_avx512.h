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

#ifndef UNITARY_CALCULATOR_AVX512_H_
#define UNITARY_CALCULATOR_AVX512_H_

#include <immintrin.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "unitaryspace_avx512.h"

namespace qsim {
namespace unitary {

/**
 * Quantum circuit unitary calculator with AVX512 vectorization.
 */
template <typename For>
class UnitaryCalculatorAVX512 final : public SimulatorBase {
 public:
  using UnitarySpace = UnitarySpaceAVX512<For>;
  using Unitary = typename UnitarySpace::Unitary;
  using fp_type = typename UnitarySpace::fp_type;

  using StateSpace = UnitarySpace;
  using State = Unitary;

  template <typename... ForArgs>
  explicit UnitaryCalculatorAVX512(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using AVX512 instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 3) {
        ApplyGateH<1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 1>(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 3) {
        ApplyGateH<2>(qs, matrix, state);
      } else if (qs[1] > 3) {
        ApplyGateL<1, 1>(qs, matrix, state);
      } else {
        ApplyGateL<0, 2>(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 3) {
        ApplyGateH<3>(qs, matrix, state);
      } else if (qs[1] > 3) {
        ApplyGateL<2, 1>(qs, matrix, state);
      } else if (qs[2] > 3) {
        ApplyGateL<1, 2>(qs, matrix, state);
      } else {
        ApplyGateL<0, 3>(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 3) {
        ApplyGateH<4>(qs, matrix, state);
      } else if (qs[1] > 3) {
        ApplyGateL<3, 1>(qs, matrix, state);
      } else if (qs[2] > 3) {
        ApplyGateL<2, 2>(qs, matrix, state);
      } else if (qs[3] > 3) {
        ApplyGateL<1, 3>(qs, matrix, state);
      } else {
        ApplyGateL<0, 4>(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 3) {
        ApplyGateH<5>(qs, matrix, state);
      } else if (qs[1] > 3) {
        ApplyGateL<4, 1>(qs, matrix, state);
      } else if (qs[2] > 3) {
        ApplyGateL<3, 2>(qs, matrix, state);
      } else if (qs[3] > 3) {
        ApplyGateL<2, 3>(qs, matrix, state);
      } else {
        ApplyGateL<1, 4>(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 3) {
        ApplyGateH<6>(qs, matrix, state);
      } else if (qs[1] > 3) {
        ApplyGateL<5, 1>(qs, matrix, state);
      } else if (qs[2] > 3) {
        ApplyGateL<4, 2>(qs, matrix, state);
      } else if (qs[3] > 3) {
        ApplyGateL<3, 3>(qs, matrix, state);
      } else {
        ApplyGateL<2, 4>(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using AVX512 instructions.
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
      if (qs[0] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<1>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 3) {
          ApplyControlledGateL<0, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 2:
      if (qs[0] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<2>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<1, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 3) {
          ApplyControlledGateL<0, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 3:
      if (qs[0] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<3>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<2, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[2] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<1, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 3) {
          ApplyControlledGateL<0, 3, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 3, 0>(qs, cqs, cvals, matrix, state);
        }
      }
      break;
    case 4:
      if (qs[0] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateHL<4>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[1] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<3, 1, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<3, 1, 0>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[2] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<2, 2, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<2, 2, 0>(qs, cqs, cvals, matrix, state);
        }
      } else if (qs[3] > 3) {
        if (cqs[0] > 3) {
          ApplyControlledGateL<1, 3, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<1, 3, 0>(qs, cqs, cvals, matrix, state);
        }
      } else {
        if (cqs[0] > 3) {
          ApplyControlledGateL<0, 4, 1>(qs, cqs, cvals, matrix, state);
        } else {
          ApplyControlledGateL<0, 4, 0>(qs, cqs, cvals, matrix, state);
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
    return 16;
  }

 private:
  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                uint64_t imaskh, uint64_t qmaskh, uint64_t size,
                uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m512 ru, iu, rn, in;
      __m512 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + _pdep_u64(r, imaskh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm512_load_ps(p0 + p);
        is[k] = _mm512_load_ps(p0 + p + 16);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm512_set1_ps(v[j]);
        iu = _mm512_set1_ps(v[j + 1]);
        rn = _mm512_mul_ps(rs[0], ru);
        in = _mm512_mul_ps(rs[0], iu);
        rn = _mm512_fnmadd_ps(is[0], iu, rn);
        in = _mm512_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm512_set1_ps(v[j]);
          iu = _mm512_set1_ps(v[j + 1]);
          rn = _mm512_fmadd_ps(rs[l], ru, rn);
          in = _mm512_fmadd_ps(rs[l], iu, in);
          rn = _mm512_fnmadd_ps(is[l], iu, rn);
          in = _mm512_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm512_store_ps(p0 + p, rn);
        _mm512_store_ps(p0 + p + 16, in);
      }
    };

    auto m = GetMasks1<H, 4>(qs);

    unsigned k = 4 + H;
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
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m512* w,
                uint64_t imaskh, uint64_t qmaskh, const __m512i* idx,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m512 rn, in;
      __m512 rs[gsize], is[gsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + _pdep_u64(r, imaskh);

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k2] = _mm512_load_ps(p0 + p);
        is[k2] = _mm512_load_ps(p0 + p + 16);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm512_permutexvar_ps(idx[l - 1], rs[k2]);
          is[k2 + l] = _mm512_permutexvar_ps(idx[l - 1], is[k2]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm512_mul_ps(rs[0], w[j]);
        in = _mm512_mul_ps(rs[0], w[j + 1]);
        rn = _mm512_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm512_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm512_fmadd_ps(rs[l], w[j], rn);
          in = _mm512_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm512_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm512_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm512_store_ps(p0 + p, rn);
        _mm512_store_ps(p0 + p + 16, in);
      }
    };

    __m512i idx[1 << L];
    __m512 w[1 << (1 + 2 * H + L)];

    auto m = GetMasks2<H, L, 4>(qs);
    FillPermutationIndices<L>(m.qmaskl, idx);
    FillMatrix<H, L, 4>(m.qmaskl, matrix, (fp_type*) w);

    unsigned k = 4 + H;
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

      __m512 ru, iu, rn, in;
      __m512 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm512_load_ps(p0 + p);
        is[k] = _mm512_load_ps(p0 + p + 16);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        ru = _mm512_set1_ps(v[j]);
        iu = _mm512_set1_ps(v[j + 1]);
        rn = _mm512_mul_ps(rs[0], ru);
        in = _mm512_mul_ps(rs[0], iu);
        rn = _mm512_fnmadd_ps(is[0], iu, rn);
        in = _mm512_fmadd_ps(is[0], ru, in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          ru = _mm512_set1_ps(v[j]);
          iu = _mm512_set1_ps(v[j + 1]);
          rn = _mm512_fmadd_ps(rs[l], ru, rn);
          in = _mm512_fmadd_ps(rs[l], iu, in);
          rn = _mm512_fnmadd_ps(is[l], iu, rn);
          in = _mm512_fmadd_ps(is[l], ru, in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm512_store_ps(p0 + p, rn);
        _mm512_store_ps(p0 + p + 16, in);
      }
    };

    auto m = GetMasks3<H, 4>(state.num_qubits(), qs, cqs, cvals);

    unsigned k = 4 + H + cqs.size();
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
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m512* w,
                uint64_t imaskh, uint64_t qmaskh, uint64_t cvalsh,
                uint64_t size, uint64_t row_size, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      __m512 rn, in;
      __m512 rs[hsize], is[hsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k] = _mm512_load_ps(p0 + p);
        is[k] = _mm512_load_ps(p0 + p + 16);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm512_mul_ps(rs[0], w[j]);
        in = _mm512_mul_ps(rs[0], w[j + 1]);
        rn = _mm512_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm512_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn = _mm512_fmadd_ps(rs[l], w[j], rn);
          in = _mm512_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm512_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm512_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm512_store_ps(p0 + p, rn);
        _mm512_store_ps(p0 + p + 16, in);
      }
    };

    __m512 w[1 << (1 + 2 * H)];

    auto m = GetMasks4<H, 4>(state.num_qubits(), qs, cqs, cvals);
    FillControlledMatrixH<H, 4>(m.cvalsl, m.cmaskl, matrix, (fp_type*) w);

    unsigned k = 4 + H + cqs.size() - m.cl;
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
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m512* w,
                uint64_t imaskh, uint64_t qmaskh, uint64_t cvalsh,
                const __m512i* idx, uint64_t size, uint64_t row_size,
                fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m512 rn, in;
      __m512 rs[gsize], is[gsize];

      uint64_t r = i % size;
      uint64_t s = i / size;

      auto p0 = rstate + row_size * s + (_pdep_u64(r, imaskh) | cvalsh);

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;
        uint64_t p = _pdep_u64(k, qmaskh);

        rs[k2] = _mm512_load_ps(p0 + p);
        is[k2] = _mm512_load_ps(p0 + p + 16);

        for (unsigned l = 1; l < lsize; ++l) {
          rs[k2 + l] = _mm512_permutexvar_ps(idx[l - 1], rs[k2]);
          is[k2 + l] = _mm512_permutexvar_ps(idx[l - 1], is[k2]);
        }
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm512_mul_ps(rs[0], w[j]);
        in = _mm512_mul_ps(rs[0], w[j + 1]);
        rn = _mm512_fnmadd_ps(is[0], w[j + 1], rn);
        in = _mm512_fmadd_ps(is[0], w[j], in);

        j += 2;

        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm512_fmadd_ps(rs[l], w[j], rn);
          in = _mm512_fmadd_ps(rs[l], w[j + 1], in);
          rn = _mm512_fnmadd_ps(is[l], w[j + 1], rn);
          in = _mm512_fmadd_ps(is[l], w[j], in);

          j += 2;
        }

        uint64_t p = _pdep_u64(k, qmaskh);

        _mm512_store_ps(p0 + p, rn);
        _mm512_store_ps(p0 + p + 16, in);
      }
    };

    __m512i idx[1 << L];
    __m512 w[1 << (1 + 2 * H + L)];

    uint64_t size2 = uint64_t{1} << state.num_qubits();
    uint64_t raw_size = UnitarySpace::MinRowSize(state.num_qubits());

    if (CH) {
      auto m = GetMasks5<H, L, 4>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillMatrix<H, L, 4>(m.qmaskl, matrix, (fp_type*) w);

      unsigned k = 4 + H + cqs.size();
      unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size * size2, f, w, m.imaskh, m.qmaskh,
               m.cvalsh, idx, size, raw_size, state.get());
    } else {
      auto m = GetMasks6<H, L, 4>(state.num_qubits(), qs, cqs, cvals);
      FillPermutationIndices<L>(m.qmaskl, idx);
      FillControlledMatrixL<H, L, 4>(
          m.cvalsl, m.cmaskl, m.qmaskl, matrix, (fp_type*) w);

      unsigned k = 4 + H + cqs.size() - m.cl;
      unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size * size2, f, w, m.imaskh, m.qmaskh,
               m.cvalsh, idx, size, raw_size, state.get());
    }
  }

  template <unsigned L>
  static void FillPermutationIndices(unsigned qmaskl, __m512i* idx) {
    constexpr unsigned lsize = 1 << L;

    for (unsigned i = 0; i < lsize; ++i) {
      unsigned p[16];

      for (unsigned j = 0; j < 16; ++j) {
        p[j] = MaskedAdd<4>(j, i + 1, qmaskl, lsize) | (j & (-1 ^ qmaskl));
      }

      idx[i] = _mm512_set_epi32(p[15], p[14], p[13], p[12], p[11], p[10],
                                p[9], p[8], p[7], p[6], p[5], p[4],
                                p[3], p[2], p[1], p[0]);
    }
  }

  For for_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_AVX512_H_
