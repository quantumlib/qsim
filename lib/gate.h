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

#ifndef GATE_H_
#define GATE_H_

#include <array>
#include <vector>

namespace qsim {

enum GateKind {
  kGateId1 = 0, // one-qubit Id
  kGateHd,      // Hadamard
  kGateT,       // T
  kGateX,       // X
  kGateY,       // Y
  kGateZ,       // Z
  kGateX2,      // sqrt(X)
  kGateY2,      // sqrt(Y)
  kGateRX,      // X-rotation
  kGateRY,      // Y-rotation
  kGateRZ,      // Z-rotation
  kGateRXY,     // XY-rotation (rotation around arbitrary axis in the XY plane)
  kGateHZ2,     // pi / 2 rotation around the X + Y axis
  kGateS,       // S
  kGateId2,     // two-qubit Id
  kGateCZ,      // CZ
  kGateCNot,    // CNOT (CX)
  kGateIS,      // iSwap
  kGateFS,      // fSim
  kGateCP,      // control phase
  kGateDecomp,  // single qubit gate from Schmidt decomposition
};

template <typename fp_type>
struct Gate {
  GateKind kind;
  unsigned time;
  unsigned num_qubits;
  unsigned qubits[3];
  bool unfusible;      // If true, the gate is fused as a master.
  bool inverse;        // If true, the qubit order was inversed (q0 > q1).
  std::vector<fp_type> params;
  std::array<fp_type, 32> matrix;
};

}  // namespace qsim

#endif  // GATE_H_
