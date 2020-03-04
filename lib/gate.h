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

enum GateAnyKind {
  kGateAny = -1,
};

template <typename fp_type, typename GK = GateAnyKind>
struct Gate {
  using GateKind = GK;

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
