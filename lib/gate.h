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

// Gate is a generic gate to make it easier to use qsim with external gate sets.
// Internal gate set is implemented in gates_def.h.
template <typename fp_type, typename GK = GateAnyKind>
struct Gate {
  using GateKind = GK;

  GateKind kind;
  unsigned time;
  unsigned num_qubits;
  unsigned qubits[3];
  bool unfusible;      // If true, the gate is fused as a master.
  bool inverse;        // If true, the qubit order is inversed (q0 > q1).
  std::vector<fp_type> params;
  std::array<fp_type, 32> matrix;
};

template <typename fp_type>
using Matrix1q = std::array<fp_type, 8>;

template <typename fp_type>
using Matrix2q = std::array<fp_type, 32>;

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::array<Matrix1q<fp_type>, 2>>;

template <typename fp_type, typename GateKind>
schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    GateKind kind, const std::vector<fp_type>& params);

}  // namespace qsim

#endif  // GATE_H_
