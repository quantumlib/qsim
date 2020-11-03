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
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, Unitary& state) const {
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
    const uint64_t sizei = uint64_t(1) << num_qubits_;
    const uint64_t sizek = uint64_t(1) << q0;

    auto data = state.get();

    auto f = [](unsigned n, unsigned m, uint64_t l, unsigned num_qubits,
                uint64_t sizei, uint64_t sizek,
                const fp_type* matrix, fp_type* data) {
      uint64_t l1 = l >> (num_qubits - 1);
      uint64_t k = l1 & (sizek - 1);
      uint64_t i = 2 * (l1 ^ k);

      uint64_t l2 = l & (sizei / 2 - 1);
      uint64_t kk = l2 & (sizek - 1);
      uint64_t ii = 2 * (l2 ^ kk);

      uint64_t si = i | k;
      uint64_t si2 = ii | kk;
      uint64_t p = si;
      uint64_t pp = si2;
      fp_type m00_r = data[2 * p * sizei + 2 * pp];
      fp_type m00_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m01_r = data[2 * p * sizei + 2 * pp];
      fp_type m01_i = data[2 * p * sizei + 2 * pp + 1];

      p = si | sizek;
      pp = si2;
      fp_type m10_r = data[2 * p * sizei + 2 * pp];
      fp_type m10_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m11_r = data[2 * p * sizei + 2 * pp];
      fp_type m11_i = data[2 * p * sizei + 2 * pp + 1];

      // End of extraction. Begin computation.
      p = si;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_one_r(m00_r, m00_i, m10_r, m10_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_one_i(m00_r, m00_i, m10_r, m10_i, 0, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_one_r(m01_r, m01_i, m11_r, m11_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_one_i(m01_r, m01_i, m11_r, m11_i, 0, matrix);

      p = si | sizek;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_one_r(m00_r, m00_i, m10_r, m10_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_one_i(m00_r, m00_i, m10_r, m10_i, 1, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_one_r(m01_r, m01_i, m11_r, m11_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_one_i(m01_r, m01_i, m11_r, m11_i, 1, matrix);
    };

    for_.Run((sizei / 2) * (sizei / 2), f,
             num_qubits_, sizei, sizek, matrix, data);
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
    const uint64_t sizei = uint64_t(1) << num_qubits_;
    const uint64_t sizej = uint64_t(1) << q1;
    const uint64_t sizek = uint64_t(1) << q0;

    auto data = state.get();

    auto f = [](unsigned n, unsigned m, uint64_t l, unsigned num_qubits,
                uint64_t sizei, uint64_t sizej, uint64_t sizek,
                const fp_type* matrix, fp_type* data) {
      uint64_t l1 = l >> (num_qubits - 2);
      uint64_t k = l1 & (sizek - 1);
      uint64_t t = l1 & (sizej / 2 - 1);
      uint64_t j = 2 * (t ^ k);
      uint64_t i = 4 * (l1 ^ t);

      uint64_t l2 = l & (sizei / 4 - 1);
      uint64_t kk = l2 & (sizek - 1);
      uint64_t tt = l2 & (sizej / 2 - 1);
      uint64_t jj = 2 * (tt ^ kk);
      uint64_t ii = 4 * (l2 ^ tt);

      uint64_t si = i | j | k;
      uint64_t si2 = ii | jj | kk;
      uint64_t p = si;
      uint64_t pp = si2;
      fp_type m00_r = data[2 * p * sizei + 2 * pp];
      fp_type m00_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m01_r = data[2 * p * sizei + 2 * pp];
      fp_type m01_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizej;
      fp_type m02_r = data[2 * p * sizei + 2 * pp];
      fp_type m02_i = data[2 * p * sizei + 2 * pp + 1];
      pp |= sizek;
      fp_type m03_r = data[2 * p * sizei + 2 * pp];
      fp_type m03_i = data[2 * p * sizei + 2 * pp + 1];

      p = si | sizek;
      pp = si2;
      fp_type m10_r = data[2 * p * sizei + 2 * pp];
      fp_type m10_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m11_r = data[2 * p * sizei + 2 * pp];
      fp_type m11_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizej;
      fp_type m12_r = data[2 * p * sizei + 2 * pp];
      fp_type m12_i = data[2 * p * sizei + 2 * pp + 1];
      pp |= sizek;
      fp_type m13_r = data[2 * p * sizei + 2 * pp];
      fp_type m13_i = data[2 * p * sizei + 2 * pp + 1];

      p = si | sizej;
      pp = si2;
      fp_type m20_r = data[2 * p * sizei + 2 * pp];
      fp_type m20_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m21_r = data[2 * p * sizei + 2 * pp];
      fp_type m21_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizej;
      fp_type m22_r = data[2 * p * sizei + 2 * pp];
      fp_type m22_i = data[2 * p * sizei + 2 * pp + 1];
      pp |= sizek;
      fp_type m23_r = data[2 * p * sizei + 2 * pp];
      fp_type m23_i = data[2 * p * sizei + 2 * pp + 1];

      p |= sizek;
      pp = si2;
      fp_type m30_r = data[2 * p * sizei + 2 * pp];
      fp_type m30_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizek;
      fp_type m31_r = data[2 * p * sizei + 2 * pp];
      fp_type m31_i = data[2 * p * sizei + 2 * pp + 1];
      pp = si2 | sizej;
      fp_type m32_r = data[2 * p * sizei + 2 * pp];
      fp_type m32_i = data[2 * p * sizei + 2 * pp + 1];
      pp |= sizek;
      fp_type m33_r = data[2 * p * sizei + 2 * pp];
      fp_type m33_i = data[2 * p * sizei + 2 * pp + 1];

      // End of extraction. Begin computation.
      p = si;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 0, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 0, matrix);
      pp = si2 | sizej;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 0, matrix);
      pp |= sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 0, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 0, matrix);

      p = si | sizek;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 1, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 1, matrix);
      pp = si2 | sizej;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 1, matrix);
      pp |= sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 1, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 1, matrix);

      p = si | sizej;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 2, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 2, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 2, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 2, matrix);
      pp = si2 | sizej;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 2, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 2, matrix);
      pp |= sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 2, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 2, matrix);

      p |= sizek;
      pp = si2;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 3, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                    m30_i, 3, matrix);
      pp = si2 | sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 3, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                    m31_i, 3, matrix);
      pp = si2 | sizej;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 3, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                    m32_i, 3, matrix);
      pp |= sizek;
      data[2 * p * sizei + 2 * pp] =
          dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 3, matrix);
      data[2 * p * sizei + 2 * pp + 1] =
          dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                    m33_i, 3, matrix);
    };

    for_.Run((sizei / 4) * (sizei / 4), f,
             num_qubits_, sizei, sizej, sizek, matrix, data);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_BASIC_H_
