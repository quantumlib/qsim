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

#ifndef SIMULATOR_AVX_H_
#define SIMULATOR_AVX_H_

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

#include "statespace_avx.h"

namespace qsim {

// Quantum circuit simulator with AVX vectorization.
template <typename For>
class SimulatorAVX final {
 public:
  using StateSpace = StateSpaceAVX<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorAVX(unsigned num_qubits, ForArgs&&... args)
      : for_(args...), num_qubits_(num_qubits) {}

  /**
   * Applies a single-qubit gate using AVX instructions.
   * @param q0 Index of the qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate1(unsigned q0, const fp_type* matrix, State& state) const {
    if (q0 > 2) {
      ApplyGate1H(q0, matrix, state);
    } else {
      ApplyGate1L(q0, matrix, state);
    }
  }

  /**
   * Applies a two-qubit gate using AVX instructions.
   * Note that qubit order is inverted in this operation.
   * @param q0 Index of the second qubit affected by this gate.
   * @param q1 Index of the first qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate2(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    // Assume q0 < q1.

    if (q0 > 2) {
      ApplyGate2HH(q0, q1, matrix, state);
    } else if (q1 > 2) {
      ApplyGate2HL(q0, q1, matrix, state);
    } else {
      ApplyGate2LL(q0, q1, matrix, state);
    }
  }

 private:
  // Applies a single-qubit gate for qubit > 2.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs AVX vectorization.
  void ApplyGate1H(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizek - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizek, uint64_t mask0, uint64_t mask1,
                const float* matrix, fp_type* rstate) {
      __m256 r0, i0, r1, i1, ru, iu, rn, in;

      uint64_t si = (32 * i & mask1) | (16 * i & mask0);
      auto p0 = rstate + si;
      auto p1 = p0 + sizek;

      r0 = _mm256_load_ps(p0);
      i0 = _mm256_load_ps(p0 + 8);
      r1 = _mm256_load_ps(p1);
      i1 = _mm256_load_ps(p1 + 8);

      ru = _mm256_set1_ps(matrix[0]);
      iu = _mm256_set1_ps(matrix[1]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[2]);
      iu = _mm256_set1_ps(matrix[3]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);

      _mm256_store_ps(p0, rn);
      _mm256_store_ps(p0 + 8, in);

      ru = _mm256_set1_ps(matrix[4]);
      iu = _mm256_set1_ps(matrix[5]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[6]);
      iu = _mm256_set1_ps(matrix[7]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);

      _mm256_store_ps(p1, rn);
      _mm256_store_ps(p1 + 8, in);
    };

    for_.Run(sizei / 16, f, sizek, mask0, mask1, matrix, rstate);
  }

  // Applies a single-qubit gate for qubit <= 2.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs AVX vectorization with permutations.
  void ApplyGate1L(unsigned q0, const fp_type* matrix, State& state) const {
    __m256i ml;
    __m256 u[4];

    switch (q0) {
    case 0:
      ml = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      u[0] = SetPs(matrix, 6, 0, 6, 0, 6, 0, 6, 0);
      u[1] = SetPs(matrix, 7, 1, 7, 1, 7, 1, 7, 1);
      u[2] = SetPs(matrix, 4, 2, 4, 2, 4, 2, 4, 2);
      u[3] = SetPs(matrix, 5, 3, 5, 3, 5, 3, 5, 3);
      break;
    case 1:
      ml = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      u[0] = SetPs(matrix, 6, 6, 0, 0, 6, 6, 0, 0);
      u[1] = SetPs(matrix, 7, 7, 1, 1, 7, 7, 1, 1);
      u[2] = SetPs(matrix, 4, 4, 2, 2, 4, 4, 2, 2);
      u[3] = SetPs(matrix, 5, 5, 3, 3, 5, 5, 3, 3);
      break;
    case 2:
      ml = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      u[0] = SetPs(matrix, 6, 6, 6, 6, 0, 0, 0, 0);
      u[1] = SetPs(matrix, 7, 7, 7, 7, 1, 1, 1, 1);
      u[2] = SetPs(matrix, 4, 4, 4, 4, 2, 2, 2, 2);
      u[3] = SetPs(matrix, 5, 5, 5, 5, 3, 3, 3, 3);
      break;
    default:
      // Cannot reach here.
      ml = _mm256_set1_epi32(0);
      for (std::size_t i = 0; i < 4; ++i) {
        u[i] = _mm256_set1_ps(0);
      }
      break;
    }

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                __m256i ml, const __m256* u, fp_type* rstate) {
      __m256 r0, i0, r1, i1, rn, in;

      auto p0 = rstate + 16 * i;

      r0 = _mm256_load_ps(p0);
      i0 = _mm256_load_ps(p0 + 8);
      r1 = _mm256_permutevar8x32_ps(r0, ml);
      i1 = _mm256_permutevar8x32_ps(i0, ml);

      rn = _mm256_mul_ps(r0, u[0]);
      in = _mm256_mul_ps(r0, u[1]);
      rn = _mm256_fnmadd_ps(i0, u[1], rn);
      in = _mm256_fmadd_ps(i0, u[0], in);
      rn = _mm256_fmadd_ps(r1, u[2], rn);
      in = _mm256_fmadd_ps(r1, u[3], in);
      rn = _mm256_fnmadd_ps(i1, u[3], rn);
      in = _mm256_fmadd_ps(i1, u[2], in);

      _mm256_store_ps(p0, rn);
      _mm256_store_ps(p0 + 8, in);
    };

    for_.Run(std::max(uint64_t{1}, sizei / 16), f, ml, u, rstate);
  }

  // Applies two-qubit gate for qubit0 > 2 and qubit1 > 2.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs AVX vectorization.
  void ApplyGate2HH(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << (num_qubits_ - 1);
    uint64_t sizej = uint64_t{1} << (q1 + 1);
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (sizej - 1) ^ (2 * sizek - 1);
    uint64_t mask2 = (4 * sizei - 1) ^ (2 * sizej - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizej, uint64_t sizek,
                uint64_t mask0, uint64_t mask1, uint64_t mask2,
                const fp_type* matrix, fp_type* rstate) {
      __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;

      uint64_t si = (64 * i & mask2) | (32 * i & mask1) | (16 * i & mask0);
      auto p0 = rstate + si;
      auto p1 = p0 + sizek;
      auto p2 = p0 + sizej;
      auto p3 = p1 + sizej;

      r0 = _mm256_load_ps(p0);
      i0 = _mm256_load_ps(p0 + 8);
      r1 = _mm256_load_ps(p1);
      i1 = _mm256_load_ps(p1 + 8);
      r2 = _mm256_load_ps(p2);
      i2 = _mm256_load_ps(p2 + 8);
      r3 = _mm256_load_ps(p3);
      i3 = _mm256_load_ps(p3 + 8);

      ru = _mm256_set1_ps(matrix[0]);
      iu = _mm256_set1_ps(matrix[1]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[2]);
      iu = _mm256_set1_ps(matrix[3]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[4]);
      iu = _mm256_set1_ps(matrix[5]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[6]);
      iu = _mm256_set1_ps(matrix[7]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      _mm256_store_ps(p0, rn);
      _mm256_store_ps(p0 + 8, in);

      ru = _mm256_set1_ps(matrix[8]);
      iu = _mm256_set1_ps(matrix[9]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[10]);
      iu = _mm256_set1_ps(matrix[11]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[12]);
      iu = _mm256_set1_ps(matrix[13]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[14]);
      iu = _mm256_set1_ps(matrix[15]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      _mm256_store_ps(p1, rn);
      _mm256_store_ps(p1 + 8, in);

      ru = _mm256_set1_ps(matrix[16]);
      iu = _mm256_set1_ps(matrix[17]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[18]);
      iu = _mm256_set1_ps(matrix[19]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[20]);
      iu = _mm256_set1_ps(matrix[21]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[22]);
      iu = _mm256_set1_ps(matrix[23]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      _mm256_store_ps(p2, rn);
      _mm256_store_ps(p2 + 8, in);

      ru = _mm256_set1_ps(matrix[24]);
      iu = _mm256_set1_ps(matrix[25]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[26]);
      iu = _mm256_set1_ps(matrix[27]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[28]);
      iu = _mm256_set1_ps(matrix[29]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[30]);
      iu = _mm256_set1_ps(matrix[31]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      _mm256_store_ps(p3, rn);
      _mm256_store_ps(p3 + 8, in);
    };

    for_.Run(sizei / 16, f, sizej, sizek, mask0, mask1, mask2, matrix, rstate);
  }

  // Applies a two-qubit gate for qubit0 <= 2 and qubit1 > 2.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs AVX vectorization with permutations.
  void ApplyGate2HL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    __m256i ml;
    __m256 u[16];

    switch (q0) {
    case 0:
      ml = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      u[ 0] = SetPs(matrix, 10,  0, 10,  0, 10,  0, 10,  0);
      u[ 1] = SetPs(matrix, 11,  1, 11,  1, 11,  1, 11,  1);
      u[ 2] = SetPs(matrix,  8,  2,  8,  2,  8,  2,  8,  2);
      u[ 3] = SetPs(matrix,  9,  3,  9,  3,  9,  3,  9,  3);
      u[ 4] = SetPs(matrix, 14,  4, 14,  4, 14,  4, 14,  4);
      u[ 5] = SetPs(matrix, 15,  5, 15,  5, 15,  5, 15,  5);
      u[ 6] = SetPs(matrix, 12,  6, 12,  6, 12,  6, 12,  6);
      u[ 7] = SetPs(matrix, 13,  7, 13,  7, 13,  7, 13,  7);
      u[ 8] = SetPs(matrix, 26, 16, 26, 16, 26, 16, 26, 16);
      u[ 9] = SetPs(matrix, 27, 17, 27, 17, 27, 17, 27, 17);
      u[10] = SetPs(matrix, 24, 18, 24, 18, 24, 18, 24, 18);
      u[11] = SetPs(matrix, 25, 19, 25, 19, 25, 19, 25, 19);
      u[12] = SetPs(matrix, 30, 20, 30, 20, 30, 20, 30, 20);
      u[13] = SetPs(matrix, 31, 21, 31, 21, 31, 21, 31, 21);
      u[14] = SetPs(matrix, 28, 22, 28, 22, 28, 22, 28, 22);
      u[15] = SetPs(matrix, 29, 23, 29, 23, 29, 23, 29, 23);
      break;
    case 1:
      ml = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      u[ 0] = SetPs(matrix, 10, 10,  0,  0, 10, 10,  0,  0);
      u[ 1] = SetPs(matrix, 11, 11,  1,  1, 11, 11,  1,  1);
      u[ 2] = SetPs(matrix,  8,  8,  2,  2,  8,  8,  2,  2);
      u[ 3] = SetPs(matrix,  9,  9,  3,  3,  9,  9,  3,  3);
      u[ 4] = SetPs(matrix, 14, 14,  4,  4, 14, 14,  4,  4);
      u[ 5] = SetPs(matrix, 15, 15,  5,  5, 15, 15,  5,  5);
      u[ 6] = SetPs(matrix, 12, 12,  6,  6, 12, 12,  6,  6);
      u[ 7] = SetPs(matrix, 13, 13,  7,  7, 13, 13,  7,  7);
      u[ 8] = SetPs(matrix, 26, 26, 16, 16, 26, 26, 16, 16);
      u[ 9] = SetPs(matrix, 27, 27, 17, 17, 27, 27, 17, 17);
      u[10] = SetPs(matrix, 24, 24, 18, 18, 24, 24, 18, 18);
      u[11] = SetPs(matrix, 25, 25, 19, 19, 25, 25, 19, 19);
      u[12] = SetPs(matrix, 30, 30, 20, 20, 30, 30, 20, 20);
      u[13] = SetPs(matrix, 31, 31, 21, 21, 31, 31, 21, 21);
      u[14] = SetPs(matrix, 28, 28, 22, 22, 28, 28, 22, 22);
      u[15] = SetPs(matrix, 29, 29, 23, 23, 29, 29, 23, 23);
      break;
    case 2:
      ml = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      u[ 0] = SetPs(matrix, 10, 10, 10, 10,  0,  0,  0,  0);
      u[ 1] = SetPs(matrix, 11, 11, 11, 11,  1,  1,  1,  1);
      u[ 2] = SetPs(matrix,  8,  8,  8,  8,  2,  2,  2,  2);
      u[ 3] = SetPs(matrix,  9,  9,  9,  9,  3,  3,  3,  3);
      u[ 4] = SetPs(matrix, 14, 14, 14, 14,  4,  4,  4,  4);
      u[ 5] = SetPs(matrix, 15, 15, 15, 15,  5,  5,  5,  5);
      u[ 6] = SetPs(matrix, 12, 12, 12, 12,  6,  6,  6,  6);
      u[ 7] = SetPs(matrix, 13, 13, 13, 13,  7,  7,  7,  7);
      u[ 8] = SetPs(matrix, 26, 26, 26, 26, 16, 16, 16, 16);
      u[ 9] = SetPs(matrix, 27, 27, 27, 27, 17, 17, 17, 17);
      u[10] = SetPs(matrix, 24, 24, 24, 24, 18, 18, 18, 18);
      u[11] = SetPs(matrix, 25, 25, 25, 25, 19, 19, 19, 19);
      u[12] = SetPs(matrix, 30, 30, 30, 30, 20, 20, 20, 20);
      u[13] = SetPs(matrix, 31, 31, 31, 31, 21, 21, 21, 21);
      u[14] = SetPs(matrix, 28, 28, 28, 28, 22, 22, 22, 22);
      u[15] = SetPs(matrix, 29, 29, 29, 29, 23, 23, 23, 23);
      break;
    default:
      // Cannot reach here.
      ml = _mm256_set1_epi32(0);
      for (std::size_t i = 0; i < 16; ++i) {
        u[i] = _mm256_set1_ps(0);
      }
      break;
    }

    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizej = uint64_t{1} << (q1 + 1);

    uint64_t mask0 = sizej - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizej - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizej, uint64_t mask0, uint64_t mask1,
                __m256i ml, const __m256* u, fp_type* rstate) {
      __m256 r0, i0, r1, i1, r2, i2, r3, i3, rn, in;

      auto p0 = rstate + ((32 * i & mask1) | (16 * i & mask0));
      auto p2 = p0 + sizej;

      r0 = _mm256_load_ps(p0);
      i0 = _mm256_load_ps(p0 + 8);
      r1 = _mm256_permutevar8x32_ps(r0, ml);
      i1 = _mm256_permutevar8x32_ps(i0, ml);
      r2 = _mm256_load_ps(p2);
      i2 = _mm256_load_ps(p2 + 8);
      r3 = _mm256_permutevar8x32_ps(r2, ml);
      i3 = _mm256_permutevar8x32_ps(i2, ml);

      rn = _mm256_mul_ps(r0, u[0]);
      in = _mm256_mul_ps(r0, u[1]);
      rn = _mm256_fnmadd_ps(i0, u[1], rn);
      in = _mm256_fmadd_ps(i0, u[0], in);
      rn = _mm256_fmadd_ps(r1, u[2], rn);
      in = _mm256_fmadd_ps(r1, u[3], in);
      rn = _mm256_fnmadd_ps(i1, u[3], rn);
      in = _mm256_fmadd_ps(i1, u[2], in);
      rn = _mm256_fmadd_ps(r2, u[4], rn);
      in = _mm256_fmadd_ps(r2, u[5], in);
      rn = _mm256_fnmadd_ps(i2, u[5], rn);
      in = _mm256_fmadd_ps(i2, u[4], in);
      rn = _mm256_fmadd_ps(r3, u[6], rn);
      in = _mm256_fmadd_ps(r3, u[7], in);
      rn = _mm256_fnmadd_ps(i3, u[7], rn);
      in = _mm256_fmadd_ps(i3, u[6], in);

      _mm256_store_ps(p0, rn);
      _mm256_store_ps(p0 + 8, in);

      rn = _mm256_mul_ps(r0, u[8]);
      in = _mm256_mul_ps(r0, u[9]);
      rn = _mm256_fnmadd_ps(i0, u[9], rn);
      in = _mm256_fmadd_ps(i0, u[8], in);
      rn = _mm256_fmadd_ps(r1, u[10], rn);
      in = _mm256_fmadd_ps(r1, u[11], in);
      rn = _mm256_fnmadd_ps(i1, u[11], rn);
      in = _mm256_fmadd_ps(i1, u[10], in);
      rn = _mm256_fmadd_ps(r2, u[12], rn);
      in = _mm256_fmadd_ps(r2, u[13], in);
      rn = _mm256_fnmadd_ps(i2, u[13], rn);
      in = _mm256_fmadd_ps(i2, u[12], in);
      rn = _mm256_fmadd_ps(r3, u[14], rn);
      in = _mm256_fmadd_ps(r3, u[15], in);
      rn = _mm256_fnmadd_ps(i3, u[15], rn);
      in = _mm256_fmadd_ps(i3, u[14], in);

      _mm256_store_ps(p2, rn);
      _mm256_store_ps(p2 + 8, in);
    };

    for_.Run(sizei / 16, f, sizej, mask0, mask1, ml, u, rstate);
  }

  // Applies a two-qubit gate for qubit0 <= 2 and qubit1 <= 2.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs AVX vectorization with permutations.
  void ApplyGate2LL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    __m256i ml[3];
    __m256 u[8];

    switch (q0 + q1) {
    case 1:
      ml[0] = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      ml[1] = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      ml[2] = _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3);

      u[0] = SetPs(matrix, 30, 20, 10,  0, 30, 20, 10,  0);
      u[1] = SetPs(matrix, 31, 21, 11,  1, 31, 21, 11,  1);
      u[2] = SetPs(matrix, 28, 22,  8,  2, 28, 22,  8,  2);
      u[3] = SetPs(matrix, 29, 23,  9,  3, 29, 23,  9,  3);
      u[4] = SetPs(matrix, 26, 16, 14,  4, 26, 16, 14,  4);
      u[5] = SetPs(matrix, 27, 17, 15,  5, 27, 17, 15,  5);
      u[6] = SetPs(matrix, 24, 18, 12,  6, 24, 18, 12,  6);
      u[7] = SetPs(matrix, 25, 19, 13,  7, 25, 19, 13,  7);
      break;
    case 2:
      ml[0] = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      ml[1] = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      ml[2] = _mm256_set_epi32(2, 3, 0, 1, 6, 7, 4, 5);

      u[0] = SetPs(matrix, 30, 20, 30, 20, 10,  0, 10,  0);
      u[1] = SetPs(matrix, 31, 21, 31, 21, 11,  1, 11,  1);
      u[2] = SetPs(matrix, 28, 22, 28, 22,  8,  2,  8,  2);
      u[3] = SetPs(matrix, 29, 23, 29, 23,  9,  3,  9,  3);
      u[4] = SetPs(matrix, 26, 16, 26, 16, 14,  4, 14,  4);
      u[5] = SetPs(matrix, 27, 17, 27, 17, 15,  5, 15,  5);
      u[6] = SetPs(matrix, 24, 18, 24, 18, 12,  6, 12,  6);
      u[7] = SetPs(matrix, 25, 19, 25, 19, 13,  7, 13,  7);
      break;
    case 3:
      ml[0] = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      ml[1] = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      ml[2] = _mm256_set_epi32(1, 0, 3, 2, 5, 4, 7, 6);

      u[0] = SetPs(matrix, 30, 30, 20, 20, 10, 10,  0,  0);
      u[1] = SetPs(matrix, 31, 31, 21, 21, 11, 11,  1,  1);
      u[2] = SetPs(matrix, 28, 28, 22, 22,  8,  8,  2,  2);
      u[3] = SetPs(matrix, 29, 29, 23, 23,  9,  9,  3,  3);
      u[4] = SetPs(matrix, 26, 26, 16, 16, 14, 14,  4,  4);
      u[5] = SetPs(matrix, 27, 27, 17, 17, 15, 15,  5,  5);
      u[6] = SetPs(matrix, 24, 24, 18, 18, 12, 12,  6,  6);
      u[7] = SetPs(matrix, 25, 25, 19, 19, 13, 13,  7,  7);
      break;
    default:
      // Cannot reach here.
      ml[0] = _mm256_set1_epi32(0);
      ml[1] = _mm256_set1_epi32(0);
      ml[2] = _mm256_set1_epi32(0);
      for (std::size_t i = 0; i < 8; ++i) {
        u[i] = _mm256_set1_ps(0);
      }
      break;
    }

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m256i* ml, const __m256* u, fp_type* rstate) {
      __m256 r0, i0, r1, i1, r2, i2, r3, i3, rn, in;

      auto p0 = rstate + 16 * i;

      r0 = _mm256_load_ps(p0);
      i0 = _mm256_load_ps(p0 + 8);
      r1 = _mm256_permutevar8x32_ps(r0, ml[0]);
      i1 = _mm256_permutevar8x32_ps(i0, ml[0]);
      r2 = _mm256_permutevar8x32_ps(r0, ml[1]);
      i2 = _mm256_permutevar8x32_ps(i0, ml[1]);
      r3 = _mm256_permutevar8x32_ps(r0, ml[2]);
      i3 = _mm256_permutevar8x32_ps(i0, ml[2]);

      rn = _mm256_mul_ps(r0, u[0]);
      in = _mm256_mul_ps(r0, u[1]);
      rn = _mm256_fnmadd_ps(i0, u[1], rn);
      in = _mm256_fmadd_ps(i0, u[0], in);
      rn = _mm256_fmadd_ps(r1, u[2], rn);
      in = _mm256_fmadd_ps(r1, u[3], in);
      rn = _mm256_fnmadd_ps(i1, u[3], rn);
      in = _mm256_fmadd_ps(i1, u[2], in);
      rn = _mm256_fmadd_ps(r2, u[4], rn);
      in = _mm256_fmadd_ps(r2, u[5], in);
      rn = _mm256_fnmadd_ps(i2, u[5], rn);
      in = _mm256_fmadd_ps(i2, u[4], in);
      rn = _mm256_fmadd_ps(r3, u[6], rn);
      in = _mm256_fmadd_ps(r3, u[7], in);
      rn = _mm256_fnmadd_ps(i3, u[7], rn);
      in = _mm256_fmadd_ps(i3, u[6], in);

      _mm256_store_ps(p0, rn);
      _mm256_store_ps(p0 + 8, in);
    };

    for_.Run(std::max(uint64_t{1}, sizei / 16), f, ml, u, rstate);
  }

  __m256 SetPs(const fp_type* m,
               unsigned i7, unsigned i6, unsigned i5, unsigned i4,
               unsigned i3, unsigned i2, unsigned i1, unsigned i0) const {
    return
        _mm256_set_ps(m[i7], m[i6], m[i5], m[i4], m[i3], m[i2], m[i1], m[i0]);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // SIMULATOR_AVX_H_
