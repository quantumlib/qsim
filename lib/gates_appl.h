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

#ifndef GATES_APPL_H_
#define GATES_APPL_H_

#include <utility>
#include <vector>

#include "matrix.h"

namespace qsim {

// Calculate 2x2 fused gate matrix.
template <typename Gate, typename Array2>
inline void CalcMatrix2(
    unsigned q0, const std::vector<Gate*>& gates, Array2& matrix) {
  Matrix2SetId(matrix);

  for (auto pgate : gates) {
    Matrix2Multiply(pgate->matrix, matrix);
  }
}

// Calculate 4x4 fused gate matrix.
template <typename Gate, typename Array2>
inline void CalcMatrix4(unsigned q0, unsigned q1,
                        const std::vector<Gate*>& gates, Array2& matrix) {
  Matrix4SetId(matrix);

  for (auto pgate : gates) {
    if (pgate->num_qubits == 1) {
      if (pgate->qubits[0] == q0) {
        Matrix4Multiply20(pgate->matrix, matrix);
      } else if (pgate->qubits[0] == q1) {
        Matrix4Multiply21(pgate->matrix, matrix);
      }
    } else {
      Matrix4Multiply(pgate->matrix, matrix);
    }
  }
}

// Apply gate.
template <typename Gate, typename Simulator, typename State>
inline void ApplyGate(
    const Simulator& simulator, const Gate& gate, State& state) {
  typename Simulator::fp_type matrix[32];

  if (gate.num_qubits == 1) {
    std::copy(gate.matrix.begin(), gate.matrix.begin() + 8, matrix);
    simulator.ApplyGate1(gate.qubits[0], matrix, state);
  } else if (gate.num_qubits == 2) {
    std::copy(gate.matrix.begin(), gate.matrix.begin() + 32, matrix);

    // Here we should have gate.qubits[0] < gate.qubits[1].
    simulator.ApplyGate2(gate.qubits[0], gate.qubits[1], matrix, state);
  }
}

// Apply fused gate.
template <typename Gate, typename Simulator, typename State>
inline void ApplyFusedGate(
    const Simulator& simulator, const Gate& gate, State& state) {
  typename Simulator::fp_type matrix[32];

  if (gate.num_qubits == 1) {
    CalcMatrix2(gate.qubits[0], gate.gates, matrix);
    simulator.ApplyGate1(gate.qubits[0], matrix, state);
  } else if (gate.num_qubits == 2) {
    // Here we should have gate.qubits[0] < gate.qubits[1].
    unsigned q0 = gate.qubits[0];
    unsigned q1 = gate.qubits[1];

    CalcMatrix4(q0, q1, gate.gates, matrix);
    simulator.ApplyGate2(q0, q1, matrix, state);
  }
}

}  // namespace qsim

#endif  // GATES_APPL_H_
