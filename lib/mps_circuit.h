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

#ifndef MPS_CIRCUIT_H_
#define MPS_CIRCUIT_H_

#include <stdexcept>

#include "circuit.h"
#include "mps_simulator.h"
#include "operation.h"

namespace qsim {
namespace mps {

// Applies a circuit's supported unitary gates to an MPS state.
// Measurements, controlled gates, channels, and gates acting on more than
// two qubits are rejected rather than silently ignored.
template <typename For, typename FP>
bool ApplyCircuit(
    const Circuit<Operation<FP>>& circuit, MPSSimulator<For, FP>& simulator,
    typename MPSSimulator<For, FP>::State& state) {
  for (const auto& op : circuit.ops) {
    if (const auto* gate = OpGetAlternative<Gate<FP>>(op)) {
      if (gate->qubits.empty() || gate->qubits.size() > 2) {
        throw std::invalid_argument(
            "MPS circuit execution supports one- and two-qubit gates only.");
      }
      simulator.ApplyGate(gate->qubits, gate->matrix.data(), state);
      continue;
    }

    throw std::invalid_argument(
        "MPS circuit execution does not yet support controlled gates, "
        "measurements, or channels.");
  }

  return true;
}

}  // namespace mps
}  // namespace qsim

#endif  // MPS_CIRCUIT_H_
