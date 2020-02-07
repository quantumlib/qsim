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

#ifndef SIMULATOR_BASIC_H_
#define SIMULATOR_BASIC_H_

#include <cstdint>

#include "statespace_basic.h"

namespace qsim {

// Quantim circuit simulator without vectorization.
template <typename ParallelFor, typename FP>
class SimulatorBasic final {
 public:
  using StateSpace = StateSpaceBasic<ParallelFor, FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  SimulatorBasic(unsigned num_qubits, unsigned num_threads)
      : num_qubits_(num_qubits), num_threads_(num_threads) {}

  // Apply a single-qubit gate.
  // Perform a sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  void ApplyGate1(unsigned q0, const fp_type* matrix, State& state) const {
    uint64_t sizei = uint64_t{1} << num_qubits_;
    uint64_t sizek = uint64_t{1} << (q0 + 1);

    uint64_t mask0 = sizek - 1;
    uint64_t mask1 = (2 * sizei - 1) ^ (2 * sizek - 1);

    auto u = matrix;
    auto rstate = StateSpace::RawData(state);

    auto f = [](unsigned n, unsigned m, uint64_t i,
                uint64_t sizek, uint64_t mask0, uint64_t mask1,
                const fp_type* u, fp_type* rstate) {
      i *= 2;
      uint64_t si0 = (2 * i & mask1) | (i & mask0);
      uint64_t si1 = si0 | sizek;

      fp_type s0r = rstate[si0 + 0];
      fp_type s0i = rstate[si0 + 1];
      fp_type s1r = rstate[si1 + 0];
      fp_type s1i = rstate[si1 + 1];

      rstate[si0 + 0] = s0r * u[0] - s0i * u[1] + s1r * u[2] - s1i * u[3];
      rstate[si0 + 1] = s0r * u[1] + s0i * u[0] + s1r * u[3] + s1i * u[2];
      rstate[si1 + 0] = s0r * u[4] - s0i * u[5] + s1r * u[6] - s1i * u[7];
      rstate[si1 + 1] = s0r * u[5] + s0i * u[4] + s1r * u[7] + s1i * u[6];
    };

    ParallelFor::Run(num_threads_, sizei / 2, f,
                     sizek, mask0, mask1, matrix, rstate);
  }

  // Apply a two-qubit gate.
  // Perform a sparse matrix-vector multiplication.
  // The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
  // The order of qubits is inverse.
  void ApplyGate2(
      unsigned q0, unsigned q1, const fp_type* matrix, State& state) const {
    // Assume q0 < q1.

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
                const fp_type* u, fp_type* rstate) {
      i *= 2;
      uint64_t si0 = (4 * i & mask2) | (2 * i & mask1) | (i & mask0);
      uint64_t si1 = si0 | sizek;
      uint64_t si2 = si0 | sizej;
      uint64_t si3 = si1 | sizej;

      fp_type s0r = rstate[si0 + 0];
      fp_type s0i = rstate[si0 + 1];
      fp_type s1r = rstate[si1 + 0];
      fp_type s1i = rstate[si1 + 1];
      fp_type s2r = rstate[si2 + 0];
      fp_type s2i = rstate[si2 + 1];
      fp_type s3r = rstate[si3 + 0];
      fp_type s3i = rstate[si3 + 1];

      rstate[si0 + 0] = s0r * u[0] - s0i * u[1] + s1r * u[2] - s1i * u[3]
          + s2r * u[4] - s2i * u[5] + s3r * u[6] - s3i * u[7];
      rstate[si0 + 1] = s0r * u[1] + s0i * u[0] + s1r * u[3] + s1i * u[2]
          + s2r * u[5] + s2i * u[4] + s3r * u[7] + s3i * u[6];
      rstate[si1 + 0] = s0r * u[8] - s0i * u[9] + s1r * u[10] - s1i * u[11]
          + s2r * u[12] - s2i * u[13] + s3r * u[14] - s3i * u[15];
      rstate[si1 + 1] = s0r * u[9] + s0i * u[8] + s1r * u[11] + s1i * u[10]
          + s2r * u[13] + s2i * u[12] + s3r * u[15] + s3i * u[14];
      rstate[si2 + 0] = s0r * u[16] - s0i * u[17] + s1r * u[18] - s1i * u[19]
          + s2r * u[20] - s2i * u[21] + s3r * u[22] - s3i * u[23];
      rstate[si2 + 1] = s0r * u[17] + s0i * u[16] + s1r * u[19] + s1i * u[18]
          + s2r * u[21] + s2i * u[20] + s3r * u[23] + s3i * u[22];
      rstate[si3 + 0] = s0r * u[24] - s0i * u[25] + s1r * u[26] - s1i * u[27]
          + s2r * u[28] - s2i * u[29] + s3r * u[30] - s3i * u[31];
      rstate[si3 + 1] = s0r * u[25] + s0i * u[24] + s1r * u[27] + s1i * u[26]
          + s2r * u[29] + s2i * u[28] + s3r * u[31] + s3i * u[30];
    };

    ParallelFor::Run(num_threads_, sizei / 2, f,
                     sizej, sizek, mask0, mask1, mask2, matrix, rstate);
  }

 private:
  unsigned num_qubits_;
  unsigned num_threads_;
};

}  // namespace qsim

#endif  // SIMULATOR_BASIC_H_
