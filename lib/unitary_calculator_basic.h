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

#ifndef UNITARY_CALCULATOR_BASIC_H_
#define UNITARY_CALCULATOR_BASIC_H_

#include <cstdint>
#include <vector>

#include "unitaryspace_basic.h"

namespace qsim {

namespace unitary {

namespace {

template <typename FP = float>
inline const FP dot_one_r(const FP m_0i_r, const FP m_0i_i, const FP m_1i_r,
                          const FP m_1i_i, const int i, const FP* gate) {
  return m_0i_r * gate[i * 4 + 0] - m_0i_i * gate[i * 4 + 1] +
         m_1i_r * gate[i * 4 + 2] - m_1i_i * gate[i * 4 + 3];
}

template <typename FP = float>
inline const FP dot_one_i(const FP m_0i_r, const FP m_0i_i, const FP m_1i_r,
                          const FP m_1i_i, const int i, const FP* gate) {
  return m_0i_r * gate[i * 4 + 1] + m_0i_i * gate[i * 4 + 0] +
         m_1i_r * gate[i * 4 + 3] + m_1i_i * gate[i * 4 + 2];
}

template <typename FP = float>
inline const FP dot_two_r(const FP m_0i_r, const FP m_0i_i, const FP m_1i_r,
                          const FP m_1i_i, const FP m_2i_r, const FP m_2i_i,
                          const FP m_3i_r, const FP m_3i_i, const int i,
                          const FP* gate) {
  return m_0i_r * gate[i * 8 + 0] - m_0i_i * gate[i * 8 + 1] +
         m_1i_r * gate[i * 8 + 2] - m_1i_i * gate[i * 8 + 3] +
         m_2i_r * gate[i * 8 + 4] - m_2i_i * gate[i * 8 + 5] +
         m_3i_r * gate[i * 8 + 6] - m_3i_i * gate[i * 8 + 7];
}

template <typename FP = float>
inline const FP dot_two_i(const FP m_0i_r, const FP m_0i_i, const FP m_1i_r,
                          const FP m_1i_i, const FP m_2i_r, const FP m_2i_i,
                          const FP m_3i_r, const FP m_3i_i, const int i,
                          const FP* gate) {
  return m_0i_r * gate[i * 8 + 1] + m_0i_i * gate[i * 8 + 0] +
         m_1i_r * gate[i * 8 + 3] + m_1i_i * gate[i * 8 + 2] +
         m_2i_r * gate[i * 8 + 5] + m_2i_i * gate[i * 8 + 4] +
         m_3i_r * gate[i * 8 + 7] + m_3i_i * gate[i * 8 + 6];
}

}  // namespace

/**
 * Quantum circuit unitary calculator without vectorization.
 */
template <typename For, typename FP = float>
class UnitaryCalculatorBasic final {
 public:
  using UnitarySpace = UnitarySpaceBasic<For, FP>;
  using Unitary = typename UnitarySpace::Unitary;
  using fp_type = typename UnitarySpace::fp_type;

  using StateSpace = UnitarySpace;
  using State = Unitary;

  template <typename... ForArgs>
  explicit UnitaryCalculatorBasic(unsigned num_qubits, ForArgs&&... args)
      : for_(args...), num_qubits_(num_qubits) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix,
                 Unitary& state) const {
    if (qs.size() == 1) {
      ApplyGate1(qs[0], matrix, state);
    } else if (qs.size() == 2) {
      // Assume qs[0] < qs[1].
      ApplyGate2(qs[0], qs[1], matrix, state);
    }
  }

  /**
   * Applies a controlled gate using non-vectorized instructions.
   * This function is not implemented.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, Unitary& state) const {
    // Not implemented.
  }

 private:
  /**
   * Applies a single-qubit gate using non-vectorized instructions.
   * The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
   * @param q0 Index of the qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate1(unsigned q0, const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;
    ms[1] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[0] - 1);

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

    auto f = [](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                unsigned sqrt_size, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      auto row_size = uint64_t{1} << n_qb;

      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]);
      uint64_t kk = (1 * j & ms[0]) | (2 * j & ms[1]);

      auto p0 = rstate + row_size * 2 * kk + 2 * k;

      for (unsigned l = 0; l < 2; ++l) {
        for (unsigned k = 0; k < 2; ++k) {
          rs[2 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[2 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 2; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 2; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 2; ++p) {
            rn += rs[2 * p + l] * v[j] - is[2 * p + l] * v[j + 1];
            in += rs[2 * p + l] * v[j + 1] + is[2 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size * size, f, matrix, ms, xss, num_qubits_, size, rstate);
  }

  /**
   * Apply a two-qubit gate using non-vectorized instructions.
   * The inner loop (V_i = \sum_j M_ij V_j) is unrolled by hand.
   * Note that qubit order is inverted in this operation.
   * @param q0 Index of the second qubit affected by this gate.
   * @param q1 Index of the first qubit affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate2(unsigned q0, unsigned q1, const fp_type* matrix,
                  Unitary& state) const {
    // Assume q0 < q1.
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (q1 + 1);
      ms[i] = ((uint64_t{1} << q1) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[1] - 1);

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

    auto f = [q0, q1](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                      const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                      unsigned sqrt_size, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      auto row_size = uint64_t{1} << n_qb;
      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2]);
      uint64_t kk = (1 * j & ms[0]) | (2 * j & ms[1]) | (4 * j & ms[2]);

      auto p0 = rstate + row_size * 2 * kk + 2 * k;

      for (unsigned l = 0; l < 4; ++l) {
        for (unsigned k = 0; k < 4; ++k) {
          rs[4 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[4 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 4; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 4; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 4; ++p) {
            rn += rs[4 * p + l] * v[j] - is[4 * p + l] * v[j + 1];
            in += rs[4 * p + l] * v[j + 1] + is[4 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size * size, f, matrix, ms, xss, num_qubits_, size, rstate);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_BASIC_H_
