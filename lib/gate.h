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
#include <cstdint>
#include <utility>
#include <vector>

#include "matrix.h"

namespace qsim {

namespace detail {

template <typename Gate, typename GateDef>
inline void SortQubits(Gate& gate) {
  for (std::size_t i = 1; i < gate.qubits.size(); ++i) {
    if (gate.qubits[i - 1] > gate.qubits[i]) {
      if (!GateDef::symmetric) {
        auto perm = NormalToGateOrderPermutation(gate.qubits);
        MatrixShuffle(perm, gate.qubits.size(), gate.matrix);
      }

      gate.swapped = true;
      std::sort(gate.qubits.begin(), gate.qubits.end());
      break;
    }
  }
}

}  // namespace detail

enum GateAnyKind {
  kGateAny = -1,
};

/**
 * A generic gate to make it easier to use qsim with external gate sets.
 */
template <typename FP, typename GK = GateAnyKind>
struct Gate {
  using fp_type = FP;
  using GateKind = GK;

  GateKind kind;
  unsigned time;
  std::vector<unsigned> qubits;
  std::vector<unsigned> controlled_by;
  uint64_t cmask;
  std::vector<fp_type> params;
  Matrix<fp_type> matrix;
  bool unfusible;      // If true, the gate is fused as a parent.
  bool swapped;        // If true, the qubit order is inversed (q0 > q1).
};

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, unsigned q0,
                       Matrix<typename Gate::fp_type>&& matrix,
                       std::vector<typename Gate::fp_type>&& params = {}) {
  return Gate{GateDef::kind, time, {q0}, {}, 0,
              std::forward<std::vector<typename Gate::fp_type>>(params),
              std::forward<Matrix<typename Gate::fp_type>>(matrix),
              false, false};
}

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, unsigned q0, unsigned q1,
                       Matrix<typename Gate::fp_type>&& matrix,
                       std::vector<typename Gate::fp_type>&& params = {}) {
  Gate gate = {GateDef::kind, time, {q0, q1}, {}, 0,
               std::forward<std::vector<typename Gate::fp_type>>(params),
               std::forward<Matrix<typename Gate::fp_type>>(matrix),
               false, false};

  if (q0 > q1) {
    gate.swapped = true;
    std::swap(gate.qubits[0], gate.qubits[1]);
    if (!GateDef::symmetric) {
      MatrixShuffle({1, 0}, 2, gate.matrix);
    }
  }

  return gate;
}

template <typename Gate, typename GateDef>
inline Gate CreateGate(unsigned time, std::vector<unsigned>&& qubits,
                       Matrix<typename Gate::fp_type>&& matrix = {},
                       std::vector<typename Gate::fp_type>&& params = {}) {
  Gate gate = {GateDef::kind, time,
               std::forward<std::vector<unsigned>>(qubits), {}, 0,
               std::forward<std::vector<typename Gate::fp_type>>(params),
               std::forward<Matrix<typename Gate::fp_type>>(matrix),
               false, false};

  detail::SortQubits<Gate, GateDef>(gate);

  return gate;
}

template <typename Gate>
inline void MakeControlledGate(std::vector<unsigned>&& controlled_by,
                               uint64_t cmask, Gate& gate) {
  gate.controlled_by = std::forward<std::vector<unsigned>>(controlled_by);
  gate.cmask = cmask;

  std::sort(gate.controlled_by.begin(), gate.controlled_by.end());
}

template <typename Gate>
inline void MakeControlledGate(std::vector<unsigned>&& controlled_by,
                               const std::vector<unsigned>& control_values,
                               Gate& gate) {
  // Assume controlled_by.size() == control_values.size().

  uint64_t cmask = 0;
  for (std::size_t i = 0; i < control_values.size(); ++i) {
    cmask |= (control_values[i] & 1) << i;
  }

  MakeControlledGate(
      std::forward<std::vector<unsigned>>(controlled_by), cmask, gate);
}

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::vector<std::vector<fp_type>>>;

template <typename fp_type, typename GateKind>
schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    GateKind kind, const std::vector<fp_type>& params);

namespace gate {

constexpr int kDecomp = 100001;       // gate from Schmidt decomposition
constexpr int kMeasurement = 100002;  // measurement gate

/**
 * A gate that simulates measurement of one or more qubits, collapsing the
 * state vector and storing the measured results.
 */
template <typename Gate>
struct Measurement {
  using GateKind = typename Gate::GateKind;

  static constexpr GateKind kind = GateKind::kMeasurement;
  static constexpr char name[] = "m";
  static constexpr bool symmetric = true;

  static Gate Create(unsigned time, std::vector<unsigned>&& qubits) {
    return CreateGate<Gate, Measurement>(time, std::move(qubits));
  }
};

}  // namespace gate

}  // namespace qsim

#endif  // GATE_H_
