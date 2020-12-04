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

#ifndef EXPECT_H_
#define EXPECT_H_

#include <complex>

#include "fuser.h"
#include "gate_appl.h"

namespace qsim {

template <typename Gate>
struct OpString {
  std::complex<double> weight;
  std::vector<Gate> ops;
};

/**
 * Computes the expectation value of the sum of operator strings (operator
 * sequences). Operators can act on any qubits and they can be any supported
 * gates. This function uses a temporary state vector.
 * @param param Options for gate fusion.
 * @param strings Operator strings.
 * @param ss StateSpace object required to copy the state vector and compute
 *   inner products.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param state The state vector of the system.
 * @param ket Temporary state vector.
 * @return The computed expectation value.
 */
template <typename Fuser, typename Simulator, typename Gate>
std::complex<double> ExpectationValue(
    const typename Fuser::Parameter& param,
    const std::vector<OpString<Gate>>& strings,
    const typename Simulator::StateSpace& ss, const Simulator& simulator,
    const typename Simulator::State& state, typename Simulator::State& ket) {
  std::complex<double> eval = 0;

  for (const auto& str : strings) {
    if (str.ops.size() == 0) continue;

    ss.Copy(state, ket);

    if (str.ops.size() == 1) {
      const auto& op = str.ops[0];
      simulator.ApplyGate(op.qubits, op.matrix.data(), ket);
    } else {
      auto fused_gates = Fuser::FuseGates(param, state.num_qubits(), str.ops);
      if (fused_gates.size() == 0) {
        eval = 0;
        break;
      }

      for (const auto& fgate : fused_gates) {
        ApplyFusedGate(simulator, fgate, ket);
      }
    }

    eval += str.weight * ss.InnerProduct(state, ket);
  }

  return eval;
}

/**
 * Computes the expectation value of the sum of operator strings (operator
 * sequences). Operators can act on any qubits and they can be any supported
 * gates except for user-defined controlled gates. Computation is performed
 * in place. No additional memory is allocated. The operator strings should
 * act on no more than six qubits and they should be fusible into one gate.
 * @param strings Operator strings.
 * @param simulator Simulator object. Provides specific implementations for
 *   computing expectation values.
 * @param state The state of the system.
 * @return The computed expectation value.
 */
template <typename IO, typename Fuser, typename Simulator, typename Gate>
std::complex<double> ExpectationValue(
    const std::vector<OpString<Gate>>& strings,
    const Simulator& simulator, const typename Simulator::State& state) {
  std::complex<double> eval = 0;

  typename Fuser::Parameter param;
  param.max_fused_size = 6;

  for (const auto& str : strings) {
    if (str.ops.size() == 0) continue;

    if (str.ops.size() == 1) {
      const auto& op = str.ops[0];
      auto r = simulator.ExpectationValue(op.qubits, op.matrix.data(), state);
      eval += str.weight * r;
    } else {
      auto fused_gates = Fuser::FuseGates(param, state.num_qubits(), str.ops);

      if (fused_gates.size() != 1) {
        IO::errorf("too many fused gates; "
                   "cannot compute the expectation value.\n");
        eval = 0;
        break;
      }

      const auto& fgate = fused_gates[0];

      if (fgate.qubits.size() > 6) {
        IO::errorf("operator string acts on too many qubits; "
                   "cannot compute the expectation value.\n");
        eval = 0;
        break;
      }

      auto matrix = CalculateFusedMatrix<typename Simulator::fp_type>(fgate);
      auto r = simulator.ExpectationValue(fgate.qubits, matrix.data(), state);
      eval += str.weight * r;
    }
  }

  return eval;
}

}  // namespace qsim

#endif  // EXPECT_H_
