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

#ifndef QUBIT_REMAP_H_
#define QUBIT_REMAP_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "circuit.h"
#include "fuser.h"
#include "gate.h"
#include "matrix.h"
#include "operation.h"

namespace qsim {
namespace qubit_remap {

namespace detail {

using QubitMap = std::vector<unsigned>;

inline QubitMap MakeIdentityQubitMap(unsigned num_qubits) {
  QubitMap map(num_qubits);
  std::iota(map.begin(), map.end(), 0);
  return map;
}

inline const Qubits& EmptyQubits() {
  static const Qubits qubits;
  return qubits;
}

template <typename Operation>
inline const Qubits& GetControlQubits(const Operation& op) {
  using FP = OpFpType<Operation>;
  if (const auto* gate = OpGetAlternative<ControlledGate<FP>>(op)) {
    return gate->controlled_by;
  }

  return EmptyQubits();
}

template <typename Operation>
inline void AddGateQubitScores(
    const Operation& op, std::vector<uint64_t>& scores) {
  const auto& qubits = OpQubits(op);
  uint64_t target_weight = qubits.size();

  for (auto q : qubits) {
    scores[q] += target_weight;
  }

  for (auto q : GetControlQubits(op)) {
    scores[q] += 1;
  }
}

template <typename Operation>
inline QubitMap MakeCacheLocalQubitMap(
    unsigned num_qubits, const std::vector<Operation>& ops) {
  auto logical_order = MakeIdentityQubitMap(num_qubits);
  std::vector<uint64_t> scores(num_qubits, 0);

  for (const auto& op : ops) {
    AddGateQubitScores(op, scores);
  }

  std::stable_sort(logical_order.begin(), logical_order.end(),
                   [&scores](auto a, auto b) {
                     return scores[a] > scores[b];
                   });

  QubitMap logical_to_physical(num_qubits);
  for (unsigned physical = 0; physical < num_qubits; ++physical) {
    logical_to_physical[logical_order[physical]] = physical;
  }

  return logical_to_physical;
}

inline QubitMap InvertQubitMap(const QubitMap& qubit_map) {
  QubitMap inverse(qubit_map.size());

  for (unsigned i = 0; i < qubit_map.size(); ++i) {
    inverse[qubit_map[i]] = i;
  }

  return inverse;
}

inline unsigned GetLowestSetBit(uint64_t mask) {
  unsigned qubit = 0;
  while ((mask & uint64_t{1}) == 0) {
    mask >>= 1;
    ++qubit;
  }

  return qubit;
}

inline uint64_t RemapIndex(uint64_t index, const QubitMap& qubit_map) {
  if (qubit_map.empty()) {
    return index;
  }

  uint64_t remapped_index = 0;

  while (index != 0) {
    unsigned qubit = GetLowestSetBit(index);
    if (qubit < qubit_map.size()) {
      remapped_index |= uint64_t{1} << qubit_map[qubit];
    } else {
      remapped_index |= uint64_t{1} << qubit;
    }
    index &= index - 1;
  }

  return remapped_index;
}

template <typename MeasurementResult>
inline void RemapMeasurementResult(
    const QubitMap& logical_to_physical, MeasurementResult& result) {
  if (logical_to_physical.empty() || !result.valid) {
    return;
  }

  auto physical_to_logical = InvertQubitMap(logical_to_physical);
  result.mask = RemapIndex(result.mask, physical_to_logical);
  result.bits = RemapIndex(result.bits, physical_to_logical);
}

template <typename Gate>
inline void RemapTargetQubits(const QubitMap& logical_to_physical, Gate& gate) {
  for (auto& q : gate.qubits) {
    q = logical_to_physical[q];
  }

  auto perm = NormalToGateOrderPermutation(gate.qubits);
  if (!perm.empty()) {
    if (!gate.matrix.empty()) {
      MatrixShuffle(perm, gate.qubits.size(), gate.matrix);
    }

    gate.swapped = true;
    std::sort(gate.qubits.begin(), gate.qubits.end());
  }
}

template <typename Gate>
inline void RemapControlQubits(const QubitMap& logical_to_physical, Gate& gate) {
  struct Control {
    unsigned qubit;
    bool value;
  };

  std::vector<Control> controls;
  controls.reserve(gate.controlled_by.size());

  for (unsigned i = 0; i < gate.controlled_by.size(); ++i) {
    controls.push_back({
        logical_to_physical[gate.controlled_by[i]],
        static_cast<bool>((gate.cmask >> i) & 1)});
  }

  std::sort(controls.begin(), controls.end(),
            [](const Control& a, const Control& b) {
              return a.qubit < b.qubit;
            });

  gate.controlled_by.clear();
  gate.cmask = 0;

  for (unsigned i = 0; i < controls.size(); ++i) {
    gate.controlled_by.push_back(controls[i].qubit);
    gate.cmask |= uint64_t{controls[i].value} << i;
  }
}

template <typename Gate>
inline void RemapGate(const QubitMap& logical_to_physical, Gate& gate) {
  RemapTargetQubits(logical_to_physical, gate);
}

template <typename FP>
inline void RemapGate(
    const QubitMap& logical_to_physical, ControlledGate<FP>& gate) {
  RemapTargetQubits(logical_to_physical, gate);
  RemapControlQubits(logical_to_physical, gate);
}

inline void RemapQubits(
    const QubitMap& logical_to_physical, Qubits& qubits) {
  for (auto& q : qubits) {
    q = logical_to_physical[q];
  }
}

template <typename Operation>
inline void RemapOperation(
    const QubitMap& logical_to_physical, Operation& op) {
  using FP = OpFpType<Operation>;

  if (auto* gate = OpGetAlternative<ControlledGate<FP>>(op)) {
    RemapGate(logical_to_physical, *gate);
  } else if (auto* gate = OpGetAlternative<Gate<FP>>(op)) {
    RemapGate(logical_to_physical, *gate);
  } else {
    RemapQubits(logical_to_physical, OpBaseOperation(op).qubits);
  }
}

template <typename Operation>
inline void RemapCircuit(
    const QubitMap& logical_to_physical, Circuit<Operation>& circuit) {
  for (auto& op : circuit.ops) {
    RemapOperation(logical_to_physical, op);
  }
}

template <typename FusedOp>
inline void RemapCopiedMeasurements(
    const QubitMap& logical_to_physical, std::vector<FusedOp>& fused_ops) {
  for (auto& op : fused_ops) {
    if (auto* measurement = OpGetAlternative<Measurement>(op)) {
      RemapQubits(logical_to_physical, measurement->qubits);
    }
  }
}

}  // namespace detail

using QubitMap = detail::QubitMap;

inline uint64_t LogicalToPhysicalIndex(
    uint64_t logical_index, const QubitMap& logical_to_physical) {
  return detail::RemapIndex(logical_index, logical_to_physical);
}

template <typename Operation>
inline QubitMap RemapCircuit(Circuit<Operation>& circuit) {
  auto logical_to_physical =
      detail::MakeCacheLocalQubitMap(circuit.num_qubits, circuit.ops);
  detail::RemapCircuit<Operation>(logical_to_physical, circuit);
  return logical_to_physical;
}

template <typename FusedOp, typename Operation>
inline QubitMap RemapCircuit(std::vector<FusedOp>& fused_ops,
                             Circuit<Operation>& circuit) {
  auto logical_to_physical =
      detail::MakeCacheLocalQubitMap(circuit.num_qubits, fused_ops);
  detail::RemapCircuit<Operation>(logical_to_physical, circuit);
  detail::RemapCopiedMeasurements(logical_to_physical, fused_ops);
  RebuildFusedGates(fused_ops);
  return logical_to_physical;
}

}  // namespace qubit_remap
}  // namespace qsim

#endif  // QUBIT_REMAP_H_
