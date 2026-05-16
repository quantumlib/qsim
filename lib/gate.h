// Copyright 2026 Google LLC. All Rights Reserved.
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
#include <variant>
#include <vector>

#include "matrix.h"
#include "operation_base.h"

namespace qsim {

// Forward declaration of Gate.
template <typename FP>
struct Gate;

/**
 * A matrix gate controlled by a number of qubits.
 */
template <typename FP>
struct ControlledGate : public Gate<FP> {
  ControlledGate() {}

  /**
   * Constructs a controlled gate from a base matrix gate. All control values
   * are set to 1.
   * @param base The underlying matrix gate (already permuted if necessary).
   * @param controlled_by The indices of the control qubits.
   */
  template <typename Q = Qubits>
  ControlledGate(const Gate<FP>& base, Q&& controlled_by)
      : Gate<FP>{base}, controlled_by(std::forward<Q>(controlled_by)),
        cmask((uint64_t{1} << this->controlled_by.size()) - 1) {}

  /**
   * Constructs a controlled gate from a base matrix gate.
   * @param base The underlying matrix gate (already permuted if necessary).
   * @param controlled_by The indices of the control qubits.
   * @param cmask A bitmask representing the required control values (0 or 1)
   *   of the control qubits.
   */
  template <typename Q = Qubits>
  ControlledGate(const Gate<FP>& base, Q&& controlled_by, uint64_t cmask)
      : Gate<FP>{base}, controlled_by(std::forward<Q>(controlled_by)),
        cmask(cmask) {}

  /** The indices of the qubits used as controls. */
  Qubits controlled_by;
  /**
   * The required bit values for the control qubits, stored as a bitmask,
   * where the i-th bit corresponds to the i-th qubit in `controlled_by`.
   */
  uint64_t cmask;
};

/**
 * Creates a controlled gate from a matrix gate.
 * @param gate The base gate to be controlled.
 * @param controlled_by The control qubit indices.
 * @return The resulting controlled gate object.
 */
template <typename FP, typename Q = Qubits>
inline ControlledGate<FP> MakeControlledGate(
    const Gate<FP>& gate, Q&& controlled_by) {
  ControlledGate<FP> cgate{gate, std::forward<Q>(controlled_by)};
  std::sort(cgate.controlled_by.begin(), cgate.controlled_by.end());

  return cgate;
}

/**
 * Creates a controlled gate from a matrix gate.
 * @param gate The base gate to be controlled.
 * @param controlled_by The control qubit indices.
 * @param control_values The control values (0 or 1) for each control qubit.
 * @return The resulting controlled gate object.
 */
template <typename FP, typename Q = Qubits>
inline ControlledGate<FP> MakeControlledGate(
    const Gate<FP>& gate, Q&& controlled_by,
    const std::vector<unsigned>& control_values) {
  // Assume controlled_by.size() == control_values.size().

  bool sorted = true;

  for (std::size_t i = 1; i < controlled_by.size(); ++i) {
    if (controlled_by[i - 1] > controlled_by[i]) {
      sorted = false;
      break;
    }
  }

  if (sorted) {
    uint64_t cmask = 0;

    for (std::size_t i = 0; i < control_values.size(); ++i) {
      cmask |= (control_values[i] & 1) << i;
    }

    return ControlledGate<FP>{gate, std::forward<Q>(controlled_by), cmask};
  } else {
    struct ControlPair {
      unsigned q;
      unsigned v;
    };

    std::vector<ControlPair> cpairs;
    cpairs.reserve(control_values.size());

    for (std::size_t i = 0; i < control_values.size(); ++i) {
      cpairs.push_back({controlled_by[i], control_values[i]});
    }

    // Sort control qubits and control values.
    std::sort(cpairs.begin(), cpairs.end(),
              [](const ControlPair& l, const ControlPair& r) -> bool {
                return l.q < r.q;
              });

    uint64_t cmask = 0;

    Qubits controlled_by;
    controlled_by.reserve(control_values.size());

    for (std::size_t i = 0; i < cpairs.size(); ++i) {
      cmask |= (cpairs[i].v & 1) << i;
      controlled_by.push_back(cpairs[i].q);
    }

    return ControlledGate<FP>{gate, std::move(controlled_by), cmask};
  }
}

/**
 * A generic matrix gate whose action is defined by a matrix.
 */
template <typename FP>
struct Gate : public BaseOperation {
  using fp_type = FP;

  /**
   * Gate parameters (e.g., rotation angles).
   * Note: Currently utilized only in qsimh.
   */
  std::vector<fp_type> params;
  /**
   * The (not necessarily unitary) matrix representing the gate operation.
   */
  Matrix<fp_type> matrix;
  /**
   * Indicates if the qubits were swapped to ensure they are in ascending order.
   */
  bool swapped;

  template <typename Q = Qubits>
  auto ControlledBy(Q&& qubits) const {
    return MakeControlledGate(*this, std::move(qubits));
  }

  template <typename Q = Qubits>
  auto ControlledBy(Q&& qubits,
                    const std::vector<unsigned>& control_values) const {
    return MakeControlledGate(*this, std::move(qubits), control_values);
  }
};

/**
 * Represents a gate that has undergone Schmidt decomposition.
 * Note: This struct is utilized only in the qsimh hybrid simulator.
 */
template <typename FP>
struct DecomposedGate : public Gate<FP> {
  /** A pointer to the original two-qubit gate that was decomposed. */
  const Gate<FP>* parent;
  /**
   * A unique identifier used to sort decomposed gate pointers into
   * a specific execution order.
   */
  unsigned id;
};

/**
 * An operation that measures a specific set of qubits at a given time step.
 */
struct Measurement : public BaseOperation {};


/**
 * A collection of gates fused into a single operation.
 * Component gates are typically multiplied into a single unitary matrix.
 * Note for qsimh: If the collection contains a `DecomposedGate`, the
 * `matrix` remains empty during the initial fusion pass. The fused
 * matrix is computed later once the Schmidt decomposition components
 * are populated.
 */
template <typename FP>
struct FusedGate : public BaseOperation {
  using fp_type = FP;

  /** Pointer to either a standard matrix gate or a Schmidt-decomposed gate. */
  using PGate = std::variant<const Gate<fp_type>*,
                             const DecomposedGate<fp_type>*>;

  /** The primary gate that initiated this fusion block. */
  PGate parent;
  /** Ordered sequence of all component gates in this block. */
  std::vector<PGate> gates;
  /**
   * The fused matrix. May be empty if `fuse_matrix` is false or if the block
   * contains a `DecomposedGate`.
   */
  Matrix<fp_type> matrix;

  /** Returns true if the primary gate is a decomposed gate. */
  bool ParentIsDecomposed() const {
    return std::holds_alternative<const DecomposedGate<fp_type>*>(parent);
  }
};

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

/**
 * A helper function to create a matrix gate. Qubit indices are sorted,
 * and the gate matrix is permuted accordingly if the gate is non-symmetric.
 * If a permutation occurs, the gate's `swapped` field is set to true.
 * @param time The time step at which the gate is applied.
 * @param qubits The qubit indices the gate acts on.
 * @param matrix The initial gate matrix (permuted during creation if needed).
 * @param params The gate parameters (utilized in qsimh).
 * @return The resulting gate object with a canonical qubit ordering.
 */
template <typename Gate, typename GateDef, typename Q = Qubits,
          typename M = Matrix<typename Gate::fp_type>>
inline Gate CreateGate(unsigned time, Q&& qubits, M&& matrix = {},
                       std::vector<typename Gate::fp_type>&& params = {}) {
  Gate gate = {GateDef::kind, time, std::forward<Q>(qubits),
               std::move(params), std::forward<M>(matrix), false};

  switch (gate.qubits.size()) {
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

  return gate;
}

enum OtherGateKind {
  kMeasurement = 100002,
  kChannel = 100003,
};

template <typename Q = Qubits>
inline Measurement CreateMeasurement(unsigned time, Q&& qubits) {
  return Measurement{kMeasurement, time, std::forward<Q>(qubits)};
}

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::vector<std::vector<fp_type>>>;

}  // namespace qsim

#endif  // GATE_H_
