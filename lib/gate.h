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

#include <algorithm>
#include <utility>
#include <vector>

#include "matrix.h"

namespace qsim {

enum GateAnyKind {
  kGateAny = -1,
};

// Gate is a generic gate to make it easier to use qsim with external gate sets.
template <typename FP, typename GK = GateAnyKind>
struct Gate {
  using fp_type = FP;
  using GateKind = GK;

  GateKind kind;
  unsigned time;
  unsigned num_qubits;
  std::vector<unsigned> qubits;
  std::vector<fp_type> params;
  std::vector<fp_type> matrix;
  bool unfusible;      // If true, the gate is fused as a master.
  bool inverse;        // If true, the qubit order is inversed (q0 > q1).
};

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, unsigned q0,
                       std::vector<typename Gate::fp_type>&& matrix,
                       std::vector<typename Gate::fp_type>&& params = {}) {
  return Gate{GateDef::kind, time, GateDef::num_qubits, {q0},
              std::move(params), std::move(matrix), false, false};
}

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, unsigned q0, unsigned q1,
                       std::vector<typename Gate::fp_type>&& matrix,
                       std::vector<typename Gate::fp_type>&& params = {}) {
  Gate gate = {GateDef::kind, time, GateDef::num_qubits, {q0, q1},
               std::move(params), std::move(matrix), false, false};
  if (q0 > q1) {
    gate.inverse = true;
    std::swap(gate.qubits[0], gate.qubits[1]);
    Matrix4Permute(gate.matrix);
  }
  return gate;
}

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, std::vector<unsigned>&& qubits,
                       std::vector<typename Gate::fp_type>&& matrix = {},
                       std::vector<typename Gate::fp_type>&& params = {}) {
  return Gate{GateDef::kind, time, static_cast<unsigned>(qubits.size()),
              std::move(qubits), std::move(params), std::move(matrix),
              false, false};
}

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::vector<std::vector<fp_type>>>;

template <typename fp_type, typename GateKind>
schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    GateKind kind, const std::vector<fp_type>& params);

namespace gate {

constexpr int kDecomp = 100001;       // gate from Schmidt decomposition
constexpr int kMeasurement = 100002;  // measurement gate

template <typename Gate>
struct Measurement {
  using GateKind = typename Gate::GateKind;

  static constexpr GateKind kind = GateKind::kMeasurement;
  static constexpr char name[] = "m";

  static Gate Create(unsigned time, std::vector<unsigned>&& qubits) {
    return CreateGate<Gate, Measurement>(time, std::move(qubits));
  }
};

}  // namespace gate

}  // namespace qsim

#endif  // GATE_H_
