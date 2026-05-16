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

#ifndef GATE_APPL_H_
#define GATE_APPL_H_

#include <utility>
#include <vector>

#include "gate.h"
#include "matrix.h"
#include "operation_base.h"

namespace qsim {

/**
 * Applies the given operation to the simulator state. Ignores measurement
 * gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param op The operation to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Operation>
inline void ApplyGate(const Simulator& simulator, const Operation& op,
                      typename Simulator::State& state) {
  using FP = typename Simulator::fp_type;

  if (const auto* pg = OpGetAlternative<Gate<FP>>(op)) {
    simulator.ApplyGate(pg->qubits, pg->matrix.data(), state);
  } else if (const auto* pg = OpGetAlternative<FusedGate<FP>>(op)) {
    simulator.ApplyGate(pg->qubits, pg->matrix.data(), state);
  } else if (const auto* pg = OpGetAlternative<ControlledGate<FP>>(op)) {
    simulator.ApplyControlledGate(pg->qubits, pg->controlled_by,
                                  pg->cmask, pg->matrix.data(), state);
  }
}

/**
 * Applies the given operation dagger to the simulator state. If the gate
 *   matrix is unitary then this is equivalent to applying the inverse gate.
 *   Ignores measurement gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param op The operation to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Operation>
inline void ApplyGateDagger(const Simulator& simulator, const Operation& op,
                            typename Simulator::State& state) {
  using FP = typename Simulator::fp_type;

  if (const auto* pg = OpGetAlternative<Gate<FP>>(op)) {
    auto matrix = pg->matrix;
    MatrixDagger(unsigned{1} << pg->qubits.size(), matrix);
    simulator.ApplyGate(pg->qubits, matrix.data(), state);
  } else if (const auto* pg = OpGetAlternative<FusedGate<FP>>(op)) {
    auto matrix = pg->matrix;
    MatrixDagger(unsigned{1} << pg->qubits.size(), matrix);
    simulator.ApplyGate(pg->qubits, matrix.data(), state);
  } else if (const auto* pg = OpGetAlternative<ControlledGate<FP>>(op)) {
    auto matrix = pg->matrix;
    MatrixDagger(unsigned{1} << pg->qubits.size(), matrix);
    simulator.ApplyControlledGate(pg->qubits, pg->controlled_by,
                                  pg->cmask, matrix.data(), state);
  }
}

/**
 * Applies the given operation to the simulator state.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param op The operation to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @param mresults As an input parameter, this can be empty or this can
 *   contain the results of the previous measurements. If gate is a measurement
 *   gate then after a successful run, the measurement result will be added to
 *   this.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Operation, typename Rgen>
inline bool ApplyGate(
    const typename Simulator::StateSpace& state_space,
    const Simulator& simulator, const Operation& op, Rgen& rgen,
    typename Simulator::State& state,
    std::vector<typename Simulator::StateSpace::MeasurementResult>& mresults) {
  if (const auto* pg = OpGetAlternative<Measurement>(op)) {
    auto measure_result = state_space.Measure(pg->qubits, rgen, state);
    if (measure_result.valid) {
      mresults.push_back(std::move(measure_result));
    } else {
      return false;
    }
  } else {
    ApplyGate(simulator, op, state);
  }

  return true;
}

/**
 * Applies the given operation to the simulator state, discarding measurement
 *   results.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param op The operation to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Operation, typename Rgen>
inline bool ApplyGate(const typename Simulator::StateSpace& state_space,
                      const Simulator& simulator, const Operation& op,
                      Rgen& rgen, typename Simulator::State& state) {
  if (const auto* pg = OpGetAlternative<Measurement>(op)) {
    auto measure_result = state_space.Measure(pg->qubits, rgen, state);
    if (!measure_result.valid) {
      return false;
    }
  } else {
    ApplyGate(simulator, op, state);
  }

  return true;
}

}  // namespace qsim

#endif  // GATE_APPL_H_
