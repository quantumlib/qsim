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
#include <xmmintrin.h>

#include <algorithm>
#include <cstdint>

#include "statespace_sse.h"

namespace qsim {

// Quantum circuit simulator with SSE vectorization.
template <typename For>
class SimulatorSSE final {
 public:
  using StateSpace = StateSpaceSSE<For>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorSSE(unsigned num_qubits, ForArgs&&... args)
      : for_(args...), num_qubits_(num_qubits) {}

  /**
   * Applies a single-qubit gate using SSE instructions.
   * @param q0 Index of the qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate1(unsigned q0, const fp_type* matrix, State& state) const {
    if (q0 > 1) {
      ApplyGate1H(q0, matrix, state);
    } else {
      ApplyGate1L(q0, matrix, state);
    }
  }

  /**
   * Applies a two-qubit gate using SSE instructions.
   * Note that qubit order is inverted in this operation.
   * @param q0 Index of the second qubit affected by this gate.
   * @param q1 Index of the first qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate2(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    // Assume q0 < q1.

    if (q0 > 1) {
      ApplyGate2HH(q0, q1, matrix, state);
    } else if (q1 > 1) {
      ApplyGate2HL(q0, q1, matrix, state);
    } else {
      ApplyGate2LL(q0, q1, matrix, state);
    }
  }

 private:
  // Applies a single-qubit gate for qubit > 1.
  // Performs a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs SSE vectorization.
  void ApplyGate1H(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizek - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizek, uint64_t mask0, uint64_t mask1,
                const fp_type* matrix, fp_type* rstate) {
      __m128 r0, i0, r1, i1, ru, iu, rn, in;

      auto p0 = rstate + ((16 * i & mask1) | (8 * i & mask0));
      auto p1 = p0 + sizek;

      r0 = _mm_load_ps(p0);
      i0 = _mm_load_ps(p0 + 4);
      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      r1 = _mm_load_ps(p1);
      i1 = _mm_load_ps(p1 + 4);
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));

      _mm_store_ps(p0, rn);
      _mm_store_ps(p0 + 4, in);

      ru = _mm_set1_ps(matrix[4]);
      iu = _mm_set1_ps(matrix[5]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[6]);
      iu = _mm_set1_ps(matrix[7]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));

      _mm_store_ps(p1, rn);
      _mm_store_ps(p1 + 4, in);
    };

    for_.Run(sizei / 8, f, sizek, mask0, mask1, matrix, rstate);
  }

  // Applies a single-qubit gate for qubit <= 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs SSE vectorization with permutations.
  void ApplyGate1L(unsigned q0, const fp_type* matrix, State& state) const {
    __m128 u[4];

    switch (q0) {
    case 0:
      u[0] = SetPs(matrix, 6, 0, 6, 0);
      u[1] = SetPs(matrix, 7, 1, 7, 1);
      u[2] = SetPs(matrix, 4, 2, 4, 2);
      u[3] = SetPs(matrix, 5, 3, 5, 3);
      break;
    case 1:
      u[0] = SetPs(matrix, 6, 6, 0, 0);
      u[1] = SetPs(matrix, 7, 7, 1, 1);
      u[2] = SetPs(matrix, 4, 4, 2, 2);
      u[3] = SetPs(matrix, 5, 5, 3, 3);
      break;
    default:
      // Cannot reach here.
      for (std::size_t i = 0; i < 4; ++i) {
        u[i] = _mm_set1_ps(0);
      }
      break;
    }

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i, unsigned q0,
                const __m128* u, fp_type* rstate) {
      __m128 r0, i0, r1, i1, rn, in;

      auto p0 = rstate + 8 * i;

      // 177 = 0b10110001: shuffle four elements dcba -> cdab.
      //  78 = 0b01001110: shuffle four elements dcba -> badc.

      r0 = _mm_load_ps(p0);
      i0 = _mm_load_ps(p0 + 4);
      r1 = q0 == 0 ? _mm_shuffle_ps(r0, r0, 177) : _mm_shuffle_ps(r0, r0, 78);
      i1 = q0 == 0 ? _mm_shuffle_ps(i0, i0, 177) : _mm_shuffle_ps(i0, i0, 78);

      rn = _mm_mul_ps(r0, u[0]);
      in = _mm_mul_ps(r0, u[1]);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, u[1]));
      in = _mm_add_ps(in, _mm_mul_ps(i0, u[0]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, u[2]));
      in = _mm_add_ps(in, _mm_mul_ps(r1, u[3]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, u[3]));
      in = _mm_add_ps(in, _mm_mul_ps(i1, u[2]));

      _mm_store_ps(p0, rn);
      _mm_store_ps(p0 + 4, in);
    };

    for_.Run(std::max(uint64_t{1}, sizei / 8), f, q0, u, rstate);
  }

  // Applies two-qubit gate for qubit0 > 1 and qubit1 > 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs SSE vectorization.
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
      __m128 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;

      uint64_t si = (32 * i & mask2) | (16 * i & mask1) | (8 * i & mask0);
      auto p0 = rstate + si;
      auto p1 = p0 + sizek;
      auto p2 = p0 + sizej;
      auto p3 = p1 + sizej;

      r0 = _mm_load_ps(p0);
      i0 = _mm_load_ps(p0 + 4);
      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      r1 = _mm_load_ps(p1);
      i1 = _mm_load_ps(p1 + 4);
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      r2 = _mm_load_ps(p2);
      i2 = _mm_load_ps(p2 + 4);
      ru = _mm_set1_ps(matrix[4]);
      iu = _mm_set1_ps(matrix[5]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      r3 = _mm_load_ps(p3);
      i3 = _mm_load_ps(p3 + 4);
      ru = _mm_set1_ps(matrix[6]);
      iu = _mm_set1_ps(matrix[7]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));

      _mm_store_ps(p0, rn);
      _mm_store_ps(p0 + 4, in);

      ru = _mm_set1_ps(matrix[8]);
      iu = _mm_set1_ps(matrix[9]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[10]);
      iu = _mm_set1_ps(matrix[11]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[12]);
      iu = _mm_set1_ps(matrix[13]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[14]);
      iu = _mm_set1_ps(matrix[15]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));

      _mm_store_ps(p1, rn);
      _mm_store_ps(p1 + 4, in);

      ru = _mm_set1_ps(matrix[16]);
      iu = _mm_set1_ps(matrix[17]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[18]);
      iu = _mm_set1_ps(matrix[19]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[20]);
      iu = _mm_set1_ps(matrix[21]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[22]);
      iu = _mm_set1_ps(matrix[23]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));

      _mm_store_ps(p2, rn);
      _mm_store_ps(p2 + 4, in);

      ru = _mm_set1_ps(matrix[24]);
      iu = _mm_set1_ps(matrix[25]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[26]);
      iu = _mm_set1_ps(matrix[27]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[28]);
      iu = _mm_set1_ps(matrix[29]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[30]);
      iu = _mm_set1_ps(matrix[31]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));

      _mm_store_ps(p3, rn);
      _mm_store_ps(p3 + 4, in);
    };

    for_.Run(sizei / 8, f, sizej, sizek, mask0, mask1, mask2, matrix, rstate);
  }

  // Applies a two-qubit gate for qubit0 <= 1 and qubit1 > 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs SSE vectorization with permutations.
  void ApplyGate2HL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    __m128 u[16];

    switch (q0) {
    case 0:
      u[ 0] = SetPs(matrix, 10,  0, 10,  0);
      u[ 1] = SetPs(matrix, 11,  1, 11,  1);
      u[ 2] = SetPs(matrix,  8,  2,  8,  2);
      u[ 3] = SetPs(matrix,  9,  3,  9,  3);
      u[ 4] = SetPs(matrix, 14,  4, 14,  4);
      u[ 5] = SetPs(matrix, 15,  5, 15,  5);
      u[ 6] = SetPs(matrix, 12,  6, 12,  6);
      u[ 7] = SetPs(matrix, 13,  7, 13,  7);
      u[ 8] = SetPs(matrix, 26, 16, 26, 16);
      u[ 9] = SetPs(matrix, 27, 17, 27, 17);
      u[10] = SetPs(matrix, 24, 18, 24, 18);
      u[11] = SetPs(matrix, 25, 19, 25, 19);
      u[12] = SetPs(matrix, 30, 20, 30, 20);
      u[13] = SetPs(matrix, 31, 21, 31, 21);
      u[14] = SetPs(matrix, 28, 22, 28, 22);
      u[15] = SetPs(matrix, 29, 23, 29, 23);
      break;
    case 1:
      u[ 0] = SetPs(matrix, 10, 10,  0,  0);
      u[ 1] = SetPs(matrix, 11, 11,  1,  1);
      u[ 2] = SetPs(matrix,  8,  8,  2,  2);
      u[ 3] = SetPs(matrix,  9,  9,  3,  3);
      u[ 4] = SetPs(matrix, 14, 14,  4,  4);
      u[ 5] = SetPs(matrix, 15, 15,  5,  5);
      u[ 6] = SetPs(matrix, 12, 12,  6,  6);
      u[ 7] = SetPs(matrix, 13, 13,  7,  7);
      u[ 8] = SetPs(matrix, 26, 26, 16, 16);
      u[ 9] = SetPs(matrix, 27, 27, 17, 17);
      u[10] = SetPs(matrix, 24, 24, 18, 18);
      u[11] = SetPs(matrix, 25, 25, 19, 19);
      u[12] = SetPs(matrix, 30, 30, 20, 20);
      u[13] = SetPs(matrix, 31, 31, 21, 21);
      u[14] = SetPs(matrix, 28, 28, 22, 22);
      u[15] = SetPs(matrix, 29, 29, 23, 23);
      break;
    default:
      // Cannot reach here.
      for (std::size_t i = 0; i < 16; ++i) {
        u[i] = _mm_set1_ps(0);
      }
      break;
    }

    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizej = uint64_t{1} << (q1 + 1);

    uint64_t mask0 = sizej - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizej - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizej, uint64_t mask0, uint64_t mask1, unsigned q0,
                const __m128* u, fp_type* rstate) {
      __m128 r0, i0, r1, i1, r2, i2, r3, i3, rn, in;

      auto p0 = rstate + ((16 * i & mask1) | (8 * i & mask0));
      auto p1 = p0 + sizej;

      // 177 = 0b10110001: shuffle four elements dcba -> cdab.
      //  78 = 0b01001110: shuffle four elements dcba -> badc.

      r0 = _mm_load_ps(p0);
      i0 = _mm_load_ps(p0 + 4);
      r1 = q0 == 0 ? _mm_shuffle_ps(r0, r0, 177) : _mm_shuffle_ps(r0, r0, 78);
      i1 = q0 == 0 ? _mm_shuffle_ps(i0, i0, 177) : _mm_shuffle_ps(i0, i0, 78);
      r2 = _mm_load_ps(p1);
      i2 = _mm_load_ps(p1 + 4);
      r3 = q0 == 0 ? _mm_shuffle_ps(r2, r2, 177) : _mm_shuffle_ps(r2, r2, 78);
      i3 = q0 == 0 ? _mm_shuffle_ps(i2, i2, 177) : _mm_shuffle_ps(i2, i2, 78);

      rn = _mm_mul_ps(r0, u[0]);
      in = _mm_mul_ps(r0, u[1]);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, u[1]));
      in = _mm_add_ps(in, _mm_mul_ps(i0, u[0]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, u[2]));
      in = _mm_add_ps(in, _mm_mul_ps(r1, u[3]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, u[3]));
      in = _mm_add_ps(in, _mm_mul_ps(i1, u[2]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, u[4]));
      in = _mm_add_ps(in, _mm_mul_ps(r2, u[5]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, u[5]));
      in = _mm_add_ps(in, _mm_mul_ps(i2, u[4]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, u[6]));
      in = _mm_add_ps(in, _mm_mul_ps(r3, u[7]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, u[7]));
      in = _mm_add_ps(in, _mm_mul_ps(i3, u[6]));

      _mm_store_ps(p0, rn);
      _mm_store_ps(p0 + 4, in);

      rn = _mm_mul_ps(r0, u[8]);
      in = _mm_mul_ps(r0, u[9]);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, u[9]));
      in = _mm_add_ps(in, _mm_mul_ps(i0, u[8]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, u[10]));
      in = _mm_add_ps(in, _mm_mul_ps(r1, u[11]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, u[11]));
      in = _mm_add_ps(in, _mm_mul_ps(i1, u[10]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, u[12]));
      in = _mm_add_ps(in, _mm_mul_ps(r2, u[13]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, u[13]));
      in = _mm_add_ps(in, _mm_mul_ps(i2, u[12]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, u[14]));
      in = _mm_add_ps(in, _mm_mul_ps(r3, u[15]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, u[15]));
      in = _mm_add_ps(in, _mm_mul_ps(i3, u[14]));

      _mm_store_ps(p1, rn);
      _mm_store_ps(p1 + 4, in);
    };

    for_.Run(sizei / 8, f, sizej, mask0, mask1, q0, u, rstate);
  }

  // Applies a two-qubit gate for qubit0 <= 1 and qubit1 > 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs SSE vectorization with permutations.
  void ApplyGate2LL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    __m128 u[8];

    u[0] = SetPs(matrix, 30, 20, 10, 0);
    u[1] = SetPs(matrix, 31, 21, 11, 1);
    u[2] = SetPs(matrix, 24, 22, 12, 2);
    u[3] = SetPs(matrix, 25, 23, 13, 3);
    u[4] = SetPs(matrix, 26, 16, 14, 4);
    u[5] = SetPs(matrix, 27, 17, 15, 5);
    u[6] = SetPs(matrix, 28, 18,  8, 6);
    u[7] = SetPs(matrix, 29, 19,  9, 7);

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const __m128* u, fp_type* rstate) {
      __m128 r0, i0, r1, i1, r2, i2, r3, i3, rn, in;

      auto p0 = rstate + 8 * i;

      r0 = _mm_load_ps(p0);
      i0 = _mm_load_ps(p0 + 4);
      r1 = _mm_shuffle_ps(r0, r0, 57);   //  57 = 0b00111001: dcba -> adcb.
      i1 = _mm_shuffle_ps(i0, i0, 57);
      r2 = _mm_shuffle_ps(r0, r0, 78);   //  78 = 0b01001110: dcba -> badc.
      i2 = _mm_shuffle_ps(i0, i0, 78);
      r3 = _mm_shuffle_ps(r0, r0, 147);  // 147 = 0b10010011: dcba -> cbad.
      i3 = _mm_shuffle_ps(i0, i0, 147);

      rn = _mm_mul_ps(r0, u[0]);
      in = _mm_mul_ps(r0, u[1]);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, u[1]));
      in = _mm_add_ps(in, _mm_mul_ps(i0, u[0]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, u[2]));
      in = _mm_add_ps(in, _mm_mul_ps(r1, u[3]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, u[3]));
      in = _mm_add_ps(in, _mm_mul_ps(i1, u[2]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, u[4]));
      in = _mm_add_ps(in, _mm_mul_ps(r2, u[5]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, u[5]));
      in = _mm_add_ps(in, _mm_mul_ps(i2, u[4]));
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, u[6]));
      in = _mm_add_ps(in, _mm_mul_ps(r3, u[7]));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, u[7]));
      in = _mm_add_ps(in, _mm_mul_ps(i3, u[6]));

      _mm_store_ps(p0, rn);
      _mm_store_ps(p0 + 4, in);
    };

    for_.Run(sizei / 8, f, u, rstate);
  }

  __m128 SetPs(const fp_type* m,
               unsigned i3, unsigned i2, unsigned i1, unsigned i0) const {
    return
        _mm_set_ps(m[i3], m[i2], m[i1], m[i0]);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // SIMULATOR_SSE_H_
