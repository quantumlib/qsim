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

#ifndef FUSER_H_
#define FUSER_H_

#include <cstdint>
#include <vector>

#include "matrix.h"

namespace qsim {

/**
 * A collection of "fused" gates which can be multiplied together before being
 * applied to the state vector.
 */
template <typename Gate>
struct GateFused {
  /**
   * Kind of the first ("parent") gate.
   */
  typename Gate::GateKind kind;
  /**
   * The time index of the first ("parent") gate.
   */
  unsigned time;
  /**
   * A list of qubits these gates act upon. Control qubits for
   * explicitly-controlled gates are excluded from this list.
   */
  std::vector<unsigned> qubits;
  /**
   * Pointer to the first ("parent") gate.
   */
  const Gate* parent;
  /**
   * Ordered list of component gates.
   */
  std::vector<const Gate*> gates;
};

/**
 * Multiplies component gate matrices of a fused gate.
 * @param gate Fused gate.
 * @return Matrix product of component matrices.
 */
template <typename fp_type, typename FusedGate>
inline Matrix<fp_type> CalculateFusedMatrix(const FusedGate& gate) {
  Matrix<fp_type> matrix;
  MatrixIdentity(unsigned{1} << gate.qubits.size(), matrix);

  for (auto pgate : gate.gates) {
    if (gate.qubits.size() == pgate->qubits.size()) {
      MatrixMultiply(gate.qubits.size(), pgate->matrix, matrix);
    } else {
      unsigned mask = 0;

      for (auto q : pgate->qubits) {
        for (std::size_t i = 0; i < gate.qubits.size(); ++i) {
          if (q == gate.qubits[i]) {
            mask |= unsigned{1} << i;
            break;
          }
        }
      }

      MatrixMultiply(mask, pgate->qubits.size(), pgate->matrix,
                     gate.qubits.size(), matrix);
    }
  }

  return matrix;
}

}  // namespace qsim

#endif  // FUSER_H_
