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
  // Performs full SSE vectorization.
  void ApplyGate1H(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizek - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizek, uint64_t mask0, uint64_t mask1,
                const fp_type* matrix, fp_type* rstate) {
      uint64_t si = (16 * i & mask1) | (8 * i & mask0);

      __m128 r0, i0, r1, i1, ru, iu, rn, in;

      uint64_t p = si;
      r0 = _mm_load_ps(rstate + p);
      i0 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      p = si | sizek;
      r1 = _mm_load_ps(rstate + p);
      i1 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      p = si;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);

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
      p = si | sizek;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);
    };

    for_.Run(sizei / 8, f, sizek, mask0, mask1, matrix, rstate);
  }

  // Applies a single-qubit gate for qubit <= 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs partial SSE vectorization with permutations.
  void ApplyGate1L(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i, unsigned q0,
                const fp_type* matrix, fp_type* rstate) {
      __m128 r0, i0, r1, i1, ru, iu, rn, in, rm, im;

      uint64_t p = 8 * i;

      r0 = _mm_load_ps(rstate + p);
      i0 = _mm_load_ps(rstate + p + 4);

      switch (q0) {
      case 0:
        r1 = _mm_shuffle_ps(r0, r0, 49);  // 00110001
        i1 = _mm_shuffle_ps(i0, i0, 49);
        break;
      case 1:
        r1 = _mm_shuffle_ps(r0, r0, 14);  // 00001110
        i1 = _mm_shuffle_ps(i0, i0, 14);
        break;
      default:
        // Cannot reach here.
        break;
      }

      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));

      ru = _mm_set1_ps(matrix[4]);
      iu = _mm_set1_ps(matrix[5]);
      rm = _mm_mul_ps(r0, ru);
      im = _mm_mul_ps(r0, iu);
      rm = _mm_sub_ps(rm, _mm_mul_ps(i0, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[6]);
      iu = _mm_set1_ps(matrix[7]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r1, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r1, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i1, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i1, ru));

      switch (q0) {
      case 0:
        rm = _mm_shuffle_ps(rm, rm, 128);  // 10000000
        im = _mm_shuffle_ps(im, im, 128);
        rn = _mm_blend_ps(rn, rm, 10);  // 1010
        in = _mm_blend_ps(in, im, 10);
        break;
      case 1:
        rn = _mm_shuffle_ps(rn, rm, 68);  // 01000100
        in = _mm_shuffle_ps(in, im, 68);
        break;
      default:
        // Cannot reach here.
        break;
      }

      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);
    };

    for_.Run(std::max(uint64_t{1}, sizei / 8), f, q0, matrix, rstate);
  }

  // Applies two-qubit gate for qubit0 > 1 and qubit1 > 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs full SSE vectorization.
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
      uint64_t si = (32 * i & mask2) | (16 * i & mask1) | (8 * i & mask0);

      __m128 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;

      uint64_t p = si;
      r0 = _mm_load_ps(rstate + p);
      i0 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      p = si | sizek;
      r1 = _mm_load_ps(rstate + p);
      i1 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      p = si | sizej;
      r2 = _mm_load_ps(rstate + p);
      i2 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[4]);
      iu = _mm_set1_ps(matrix[5]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      p |= sizek;
      r3 = _mm_load_ps(rstate + p);
      i3 = _mm_load_ps(rstate + p + 4);
      ru = _mm_set1_ps(matrix[6]);
      iu = _mm_set1_ps(matrix[7]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));
      p = si;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);

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
      p = si | sizek;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);

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
      p = si | sizej;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);

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
      p |= sizek;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);
    };

    for_.Run(sizei / 8, f, sizej, sizek, mask0, mask1, mask2, matrix, rstate);
  }

  // Applies a two-qubit gate for qubit0 <= 1 and qubit1 > 1.
  // Performs vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Performs partial SSE vectorization with permutations.
  void ApplyGate2HL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizej = uint64_t{1} << (q1 + 1);

    uint64_t mask0 = sizej - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizej - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizej, uint64_t mask0, uint64_t mask1, unsigned q0,
                const fp_type* matrix, fp_type* rstate) {
      uint64_t si = (16 * i & mask1) | (8 * i & mask0);

      __m128 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;

      uint64_t p = si;

      r0 = _mm_load_ps(rstate + p);
      i0 = _mm_load_ps(rstate + p + 4);
      p = si | sizej;
      r2 = _mm_load_ps(rstate + p);
      i2 = _mm_load_ps(rstate + p + 4);

      switch (q0) {
      case 0:
        r1 = _mm_shuffle_ps(r0, r0, 49);  // 00110001
        i1 = _mm_shuffle_ps(i0, i0, 49);
        r3 = _mm_shuffle_ps(r2, r2, 49);
        i3 = _mm_shuffle_ps(i2, i2, 49);
        break;
      case 1:
        r1 = _mm_shuffle_ps(r0, r0, 14);  // 00001110
        i1 = _mm_shuffle_ps(i0, i0, 14);
        r3 = _mm_shuffle_ps(r2, r2, 14);
        i3 = _mm_shuffle_ps(i2, i2, 14);
        break;
      default:
        // Cannot reach here.
        break;
      }

      ru = _mm_set1_ps(matrix[0]);
      iu = _mm_set1_ps(matrix[1]);
      rn = _mm_mul_ps(r0, ru);
      in = _mm_mul_ps(r0, iu);
      rn = _mm_sub_ps(rn, _mm_mul_ps(i0, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[2]);
      iu = _mm_set1_ps(matrix[3]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r1, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r1, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i1, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[4]);
      iu = _mm_set1_ps(matrix[5]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r2, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r2, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i2, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[6]);
      iu = _mm_set1_ps(matrix[7]);
      rn = _mm_add_ps(rn, _mm_mul_ps(r3, ru));
      in = _mm_add_ps(in, _mm_mul_ps(r3, iu));
      rn = _mm_sub_ps(rn, _mm_mul_ps(i3, iu));
      in = _mm_add_ps(in, _mm_mul_ps(i3, ru));

      ru = _mm_set1_ps(matrix[8]);
      iu = _mm_set1_ps(matrix[9]);
      rm = _mm_mul_ps(r0, ru);
      im = _mm_mul_ps(r0, iu);
      rm = _mm_sub_ps(rm, _mm_mul_ps(i0, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[10]);
      iu = _mm_set1_ps(matrix[11]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r1, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r1, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i1, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[12]);
      iu = _mm_set1_ps(matrix[13]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r2, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r2, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i2, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[14]);
      iu = _mm_set1_ps(matrix[15]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r3, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r3, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i3, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i3, ru));

      switch (q0) {
      case 0:
        rm = _mm_shuffle_ps(rm, rm, 128);  // 10000000
        im = _mm_shuffle_ps(im, im, 128);
        rn = _mm_blend_ps(rn, rm, 10);  // 1010
        in = _mm_blend_ps(in, im, 10);
        break;
      case 1:
        rn = _mm_shuffle_ps(rn, rm, 68);  // 01000100
        in = _mm_shuffle_ps(in, im, 68);
        break;
      default:
        // Cannot reach here.
        break;
      }

      p = si;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);

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

      ru = _mm_set1_ps(matrix[24]);
      iu = _mm_set1_ps(matrix[25]);
      rm = _mm_mul_ps(r0, ru);
      im = _mm_mul_ps(r0, iu);
      rm = _mm_sub_ps(rm, _mm_mul_ps(i0, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i0, ru));
      ru = _mm_set1_ps(matrix[26]);
      iu = _mm_set1_ps(matrix[27]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r1, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r1, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i1, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i1, ru));
      ru = _mm_set1_ps(matrix[28]);
      iu = _mm_set1_ps(matrix[29]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r2, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r2, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i2, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i2, ru));
      ru = _mm_set1_ps(matrix[30]);
      iu = _mm_set1_ps(matrix[31]);
      rm = _mm_add_ps(rm, _mm_mul_ps(r3, ru));
      im = _mm_add_ps(im, _mm_mul_ps(r3, iu));
      rm = _mm_sub_ps(rm, _mm_mul_ps(i3, iu));
      im = _mm_add_ps(im, _mm_mul_ps(i3, ru));

      switch (q0) {
      case 0:
        rm = _mm_shuffle_ps(rm, rm, 128);  // 10000000
        im = _mm_shuffle_ps(im, im, 128);
        rn = _mm_blend_ps(rn, rm, 10);  // 1010
        in = _mm_blend_ps(in, im, 10);
        break;
      case 1:
        rn = _mm_shuffle_ps(rn, rm, 68);  // 01000100
        in = _mm_shuffle_ps(in, im, 68);
        break;
      default:
        // Cannot reach here.
        break;
      }

      p = si | sizej;
      _mm_store_ps(rstate + p, rn);
      _mm_store_ps(rstate + p + 4, in);
    };

    for_.Run(sizei / 8, f, sizej, mask0, mask1, q0, matrix, rstate);
  }

  // Applies a two-qubit gate for qubit0 = 0 and qubit1 = 1.
  // Performs sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  void ApplyGate2LL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* u,
                fp_type* rstate) {
      uint64_t p = 8 * i;

      fp_type s0r = rstate[p + 0];
      fp_type s1r = rstate[p + 1];
      fp_type s2r = rstate[p + 2];
      fp_type s3r = rstate[p + 3];
      fp_type s0i = rstate[p + 4];
      fp_type s1i = rstate[p + 5];
      fp_type s2i = rstate[p + 6];
      fp_type s3i = rstate[p + 7];

      rstate[p + 0] = s0r * u[0] - s0i * u[1] + s1r * u[2] - s1i * u[3]
          + s2r * u[4] - s2i * u[5] + s3r * u[6] - s3i * u[7];
      rstate[p + 4] = s0r * u[1] + s0i * u[0] + s1r * u[3] + s1i * u[2]
          + s2r * u[5] + s2i * u[4] + s3r * u[7] + s3i * u[6];
      rstate[p + 1] = s0r * u[8] - s0i * u[9] + s1r * u[10] - s1i * u[11]
          + s2r * u[12] - s2i * u[13] + s3r * u[14] - s3i * u[15];
      rstate[p + 5] = s0r * u[9] + s0i * u[8] + s1r * u[11] + s1i * u[10]
          + s2r * u[13] + s2i * u[12] + s3r * u[15] + s3i * u[14];
      rstate[p + 2] = s0r * u[16] - s0i * u[17] + s1r * u[18] - s1i * u[19]
          + s2r * u[20] - s2i * u[21] + s3r * u[22] - s3i * u[23];
      rstate[p + 6] = s0r * u[17] + s0i * u[16] + s1r * u[19] + s1i * u[18]
          + s2r * u[21] + s2i * u[20] + s3r * u[23] + s3i * u[22];
      rstate[p + 3] = s0r * u[24] - s0i * u[25] + s1r * u[26] - s1i * u[27]
          + s2r * u[28] - s2i * u[29] + s3r * u[30] - s3i * u[31];
      rstate[p + 7] = s0r * u[25] + s0i * u[24] + s1r * u[27] + s1i * u[26]
          + s2r * u[29] + s2i * u[28] + s3r * u[31] + s3i * u[30];
    };

    for_.Run(sizei / 8, f, matrix, rstate);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace qsim

#endif  // SIMULATOR_SSE_H_
