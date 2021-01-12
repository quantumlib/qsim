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

template <typename Qubits = std::vector<unsigned>, typename Gate>
inline void MakeControlledGate(Qubits&& controlled_by, Gate& gate) {
  gate.controlled_by = std::forward<Qubits>(controlled_by);
  gate.cmask = (uint64_t{1} << gate.controlled_by.size()) - 1;

  std::sort(gate.controlled_by.begin(), gate.controlled_by.end());
}

template <typename Qubits = std::vector<unsigned>, typename Gate>
inline void MakeControlledGate(Qubits&& controlled_by,
                               const std::vector<unsigned>& control_values,
                               Gate& gate) {
  // Assume controlled_by.size() == control_values.size().

  bool sorted = true;

  for (std::size_t i = 1; i < controlled_by.size(); ++i) {
    if (controlled_by[i - 1] > controlled_by[i]) {
      sorted = false;
      break;
    }
  }

  if (sorted) {
    gate.controlled_by = std::forward<Qubits>(controlled_by);
    gate.cmask = 0;

    for (std::size_t i = 0; i < control_values.size(); ++i) {
      gate.cmask |= (control_values[i] & 1) << i;
    }
  } else {
    struct ControlPair {
      unsigned q;
      unsigned v;
    };

    std::vector<ControlPair> cpairs;
    cpairs.reserve(controlled_by.size());

    for (std::size_t i = 0; i < controlled_by.size(); ++i) {
      cpairs.push_back({controlled_by[i], control_values[i]});
    }

    // Sort control qubits and control values.
    std::sort(cpairs.begin(), cpairs.end(),
              [](const ControlPair& l, const ControlPair& r) -> bool {
                return l.q < r.q;
              });

    gate.cmask = 0;
    gate.controlled_by.reserve(controlled_by.size());

    for (std::size_t i = 0; i < cpairs.size(); ++i) {
      gate.cmask |= (cpairs[i].v & 1) << i;
      gate.controlled_by.push_back(cpairs[i].q);
    }
  }
}

namespace gate {

constexpr int kDecomp = 100001;       // gate from Schmidt decomposition
constexpr int kMeasurement = 100002;  // measurement gate

}  // namespace gate

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
  bool swapped;        // If true, the gate qubits are swapped to make qubits
                       // ordered in ascending order. This does not apply to
                       // control qubits of explicitly-controlled gates.

  template <typename Qubits = std::vector<unsigned>>
  Gate&& ControlledBy(Qubits&& controlled_by) {
    MakeControlledGate(std::forward<Qubits>(controlled_by), *this);
    return std::move(*this);
  }

  template <typename Qubits = std::vector<unsigned>>
  Gate&& ControlledBy(Qubits&& controlled_by,
                      const std::vector<unsigned>& control_values) {
    MakeControlledGate(
        std::forward<Qubits>(controlled_by), control_values, *this);
    return std::move(*this);
  }
};

template <typename Gate, typename GateDef,
          typename Qubits = std::vector<unsigned>,
          typename M = Matrix<typename Gate::fp_type>>
inline Gate CreateGate(unsigned time, Qubits&& qubits, M&& matrix = {},
                       std::vector<typename Gate::fp_type>&& params = {}) {
  Gate gate = {GateDef::kind, time, std::forward<Qubits>(qubits), {}, 0,
               std::move(params), std::forward<M>(matrix), false, false};

  if (GateDef::kind != gate::kMeasurement) {
    switch (qubits.size()) {
    case 1:
      break;
    case 2:
      if (gate.qubits[0] > gate.qubits[1]) {
        gate.swapped = true;
        std::swap(gate.qubits[0], gate.qubits[1]);
        if (!GateDef::symmetric) {
          MatrixShuffle({1, 0}, 2, gate.matrix);
        }
      }
      break;
    default:
      detail::SortQubits<Gate, GateDef>(gate);
    }
  }

  return gate;
}

namespace gate {

/**
 * A gate that simulates measurement of one or more qubits, collapsing the
 * state vector and storing the measured results.
 */
template <typename Gate>
struct Measurement {
  using GateKind = typename Gate::GateKind;

  static constexpr GateKind kind = GateKind::kMeasurement;
  static constexpr char name[] = "m";
  static constexpr bool symmetric = false;

  template <typename Qubits = std::vector<unsigned>>
  static Gate Create(unsigned time, Qubits&& qubits) {
    return CreateGate<Gate, Measurement>(time, std::forward<Qubits>(qubits));
  }
};

}  // namespace gate

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::vector<std::vector<fp_type>>>;

template <typename fp_type, typename GateKind>
schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    GateKind kind, const std::vector<fp_type>& params);

}  // namespace qsim

#endif  // GATE_H_
