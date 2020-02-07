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

// Quantim circuit simulator with AVX vectorization.
template <typename ParallelFor>
class SimulatorAVX final {
 public:
  using StateSpace = StateSpaceAVX<ParallelFor>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  SimulatorAVX(unsigned num_qubits, unsigned num_threads)
      : num_qubits_(num_qubits), num_threads_(num_threads) {}

  // Apply a single-qubit gate.
  void ApplyGate1(unsigned q0, const fp_type* matrix, State& state) const {
    if (q0 > 2) {
      ApplyGate1H(q0, matrix, state);
    } else {
      ApplyGate1L(q0, matrix, state);
    }
  }

  // Apply a two-qubit gate.
  // The order of qubits is inverse.
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
  // Apply a single-qubit gate for qubit > 2.
  // Perform a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Perform full AVX vectorization.
  void ApplyGate1H(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizek - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizek, uint64_t mask0, uint64_t mask1,
                const fp_type* matrix, fp_type* rstate) {
      i *= 16;
      uint64_t si = (2 * i & mask1) | (i & mask0);

      __m256 r0, i0, r1, i1, ru, iu, rn, in;

      uint64_t p = si;
      r0 = _mm256_load_ps(rstate + p);
      i0 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[0]);
      iu = _mm256_set1_ps(matrix[1]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      p = si | sizek;
      r1 = _mm256_load_ps(rstate + p);
      i1 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[2]);
      iu = _mm256_set1_ps(matrix[3]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      p = si;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

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
      p = si | sizek;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);
    };

    ParallelFor::Run(num_threads_, sizei / 16, f,
                     sizek, mask0, mask1, matrix, rstate);
  }

  // Apply a single-qubit gate for qubit <= 2.
  // Perform a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Perform partial AVX vectorization with permutations.
  void ApplyGate1L(unsigned q0, const fp_type* matrix, State& state) const {
    __m256i ml;

    switch (q0) {
    case 0:
      ml = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      break;
    case 1:
      ml = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      break;
    case 2:
      ml = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      break;
    default:
      // Cannot reach here.
      break;
    }

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i, unsigned q0,
                const __m256i& ml, const fp_type* matrix, fp_type* rstate) {
      __m256 r0, i0, r1, i1, ru, iu, rn, in, rm, im;

      auto p = rstate + 16 * i;

      r0 = _mm256_load_ps(p);
      i0 = _mm256_load_ps(p + 8);

      r1 = _mm256_permutevar8x32_ps(r0, ml);
      i1 = _mm256_permutevar8x32_ps(i0, ml);

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
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[6]);
      iu = _mm256_set1_ps(matrix[7]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml);
      im = _mm256_permutevar8x32_ps(im, ml);

      switch (q0) {
      case 0:
        rn = _mm256_blend_ps(rn, rm, 170);  // 10101010
        in = _mm256_blend_ps(in, im, 170);
        break;
      case 1:
        rn = _mm256_blend_ps(rn, rm, 204);  // 11001100
        in = _mm256_blend_ps(in, im, 204);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 240);  // 11110000
        in = _mm256_blend_ps(in, im, 240);
        break;
      }

      _mm256_store_ps(p, rn);
      _mm256_store_ps(p + 8, in);
    };

    ParallelFor::Run(num_threads_, std::max(uint64_t{1}, sizei / 16), f,
                     q0, ml, matrix, rstate);
  }

  // Apply a two-qubit gate for qubit0 > 2 and qubit1 > 2.
  // Perform a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Perform full AVX vectorization.
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
      i *= 16;
      uint64_t si = (4 * i & mask2) | (2 * i & mask1) | (i & mask0);

      __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;

      uint64_t p = si;
      r0 = _mm256_load_ps(rstate + p);
      i0 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[0]);
      iu = _mm256_set1_ps(matrix[1]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      p = si | sizek;
      r1 = _mm256_load_ps(rstate + p);
      i1 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[2]);
      iu = _mm256_set1_ps(matrix[3]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      p = si | sizej;
      r2 = _mm256_load_ps(rstate + p);
      i2 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[4]);
      iu = _mm256_set1_ps(matrix[5]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      p |= sizek;
      r3 = _mm256_load_ps(rstate + p);
      i3 = _mm256_load_ps(rstate + p + 8);
      ru = _mm256_set1_ps(matrix[6]);
      iu = _mm256_set1_ps(matrix[7]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);
      p = si;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

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
      p = si | sizek;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

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
      p = si | sizej;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

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
      p |= sizek;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);
    };

    ParallelFor::Run(num_threads_, sizei / 16, f,
                     sizej, sizek, mask0, mask1, mask2, matrix, rstate);
  }

  // Apply a two-qubit gate for qubit0 <= 2 and qubit1 > 2.
  // Perform a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Perform partial AVX vectorization with permutations.
  void ApplyGate2HL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    __m256i ml;

    switch (q0) {
    case 0:
      ml = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      break;
    case 1:
      ml = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      break;
    case 2:
      ml = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      break;
    default:
      // Cannot reach here.
      break;
    }

    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizej = uint64_t{1} << (q1 + 1);

    uint64_t mask0 = sizej - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizej - 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizej, uint64_t mask0, uint64_t mask1, unsigned q0,
                const __m256i& ml, const fp_type* matrix, fp_type* rstate) {
      i *= 16;
      uint64_t si = (2 * i & mask1) | (i & mask0);

      __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;

      uint64_t p = si;

      r0 = _mm256_load_ps(rstate + p);
      i0 = _mm256_load_ps(rstate + p + 8);

      r1 = _mm256_permutevar8x32_ps(r0, ml);
      i1 = _mm256_permutevar8x32_ps(i0, ml);

      p = si | sizej;

      r2 = _mm256_load_ps(rstate + p);
      i2 = _mm256_load_ps(rstate + p + 8);

      r3 = _mm256_permutevar8x32_ps(r2, ml);
      i3 = _mm256_permutevar8x32_ps(i2, ml);

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

      ru = _mm256_set1_ps(matrix[8]);
      iu = _mm256_set1_ps(matrix[9]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[10]);
      iu = _mm256_set1_ps(matrix[11]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[12]);
      iu = _mm256_set1_ps(matrix[13]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[14]);
      iu = _mm256_set1_ps(matrix[15]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml);
      im = _mm256_permutevar8x32_ps(im, ml);

      switch (q0) {
      case 0:
        rn = _mm256_blend_ps(rn, rm, 170);  // 10101010
        in = _mm256_blend_ps(in, im, 170);
        break;
      case 1:
        rn = _mm256_blend_ps(rn, rm, 204);  // 11001100
        in = _mm256_blend_ps(in, im, 204);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 240);  // 11110000
        in = _mm256_blend_ps(in, im, 240);
        break;
      }

      p = si;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

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

      ru = _mm256_set1_ps(matrix[24]);
      iu = _mm256_set1_ps(matrix[25]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[26]);
      iu = _mm256_set1_ps(matrix[27]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[28]);
      iu = _mm256_set1_ps(matrix[29]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[30]);
      iu = _mm256_set1_ps(matrix[31]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml);
      im = _mm256_permutevar8x32_ps(im, ml);

      switch (q0) {
      case 0:
        rn = _mm256_blend_ps(rn, rm, 170);  // 10101010
        in = _mm256_blend_ps(in, im, 170);
        break;
      case 1:
        rn = _mm256_blend_ps(rn, rm, 204);  // 11001100
        in = _mm256_blend_ps(in, im, 204);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 240);  // 11110000
        in = _mm256_blend_ps(in, im, 240);
        break;
      }

      p = si | sizej;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);
    };

    ParallelFor::Run(num_threads_, sizei / 16, f,
                     sizej, mask0, mask1, q0, ml, matrix, rstate);
  }

  // Apply a two-qubit gate for qubit0 <= 2 and qubit1 <= 2.
  // Perform a vectorized sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // Perform partial AVX vectorization with permutations.
  void ApplyGate2LL(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    unsigned q = q0 + q1;

    __m256i ml1, ml2, ml3;

    switch (q) {
    case 1:
      ml1 = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
      ml2 = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
      ml3 = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
      break;
    case 2:
      ml1 = _mm256_set_epi32(7, 6, 5, 4, 2, 3, 0, 1);
      ml2 = _mm256_set_epi32(7, 2, 5, 0, 3, 6, 1, 4);
      ml3 = _mm256_set_epi32(2, 6, 0, 4, 3, 7, 1, 5);
      break;
    case 3:
      ml1 = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
      ml2 = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
      ml3 = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
      break;
    default:
      // Cannot reach here.
      break;
    }

    uint64_t sizei = uint64_t{1} << (num_qubits_ + 1);

    fp_type* rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i, unsigned q,
                const __m256i& ml1, const __m256i& ml2, const __m256i& ml3,
                const fp_type* matrix, fp_type* rstate) {
      __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;

      auto p = rstate + 16 * i;

      r0 = _mm256_load_ps(p);
      i0 = _mm256_load_ps(p + 8);

      r1 = _mm256_permutevar8x32_ps(r0, ml1);
      i1 = _mm256_permutevar8x32_ps(i0, ml1);

      r2 = _mm256_permutevar8x32_ps(r0, ml2);
      i2 = _mm256_permutevar8x32_ps(i0, ml2);

      r3 = _mm256_permutevar8x32_ps(r0, ml3);
      i3 = _mm256_permutevar8x32_ps(i0, ml3);

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

      ru = _mm256_set1_ps(matrix[8]);
      iu = _mm256_set1_ps(matrix[9]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[10]);
      iu = _mm256_set1_ps(matrix[11]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[12]);
      iu = _mm256_set1_ps(matrix[13]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[14]);
      iu = _mm256_set1_ps(matrix[15]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml1);
      im = _mm256_permutevar8x32_ps(im, ml1);

      switch (q) {
      case 1:
        rn = _mm256_blend_ps(rn, rm, 34);  // 00100010
        in = _mm256_blend_ps(in, im, 34);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 10);  // 00001010
        in = _mm256_blend_ps(in, im, 10);
        break;
      case 3:
        rn = _mm256_blend_ps(rn, rm, 12);  // 00001100
        in = _mm256_blend_ps(in, im, 12);
        break;
      }

      ru = _mm256_set1_ps(matrix[16]);
      iu = _mm256_set1_ps(matrix[17]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[18]);
      iu = _mm256_set1_ps(matrix[19]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[20]);
      iu = _mm256_set1_ps(matrix[21]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[22]);
      iu = _mm256_set1_ps(matrix[23]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml2);
      im = _mm256_permutevar8x32_ps(im, ml2);

      switch (q) {
      case 1:
        rn = _mm256_blend_ps(rn, rm, 68);  // 01000100
        in = _mm256_blend_ps(in, im, 68);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 80);  // 01010000
        in = _mm256_blend_ps(in, im, 80);
        break;
      case 3:
        rn = _mm256_blend_ps(rn, rm, 48);  // 00110000
        in = _mm256_blend_ps(in, im, 48);
        break;
      }

      ru = _mm256_set1_ps(matrix[24]);
      iu = _mm256_set1_ps(matrix[25]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[26]);
      iu = _mm256_set1_ps(matrix[27]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[28]);
      iu = _mm256_set1_ps(matrix[29]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[30]);
      iu = _mm256_set1_ps(matrix[31]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml3);
      im = _mm256_permutevar8x32_ps(im, ml3);

      switch (q) {
      case 1:
        rn = _mm256_blend_ps(rn, rm, 136);  // 10001000
        in = _mm256_blend_ps(in, im, 136);
        break;
      case 2:
        rn = _mm256_blend_ps(rn, rm, 160);  // 10100000
        in = _mm256_blend_ps(in, im, 160);
        break;
      case 3:
        rn = _mm256_blend_ps(rn, rm, 192);  // 11000000
        in = _mm256_blend_ps(in, im, 192);
        break;
      }

      _mm256_store_ps(p, rn);
      _mm256_store_ps(p + 8, in);
    };

    ParallelFor::Run(num_threads_, std::max(uint64_t{1}, sizei / 16), f,
                     q, ml1, ml2, ml3, matrix, rstate);
  }

  unsigned num_qubits_;
  unsigned num_threads_;
};

}  // namespace qsim

#endif  // SIMULATOR_AVX_H_
