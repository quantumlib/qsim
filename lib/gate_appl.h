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

#include "fuser.h"
#include "gate.h"
#include "matrix.h"

namespace qsim {

/**
 * Applies the given gate to the simulator state. Ignores measurement gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Gate>
inline void ApplyGate(const Simulator& simulator, const Gate& gate,
                      typename Simulator::State& state) {
  if (gate.kind != gate::kMeasurement) {
    if (gate.controlled_by.size() == 0) {
      simulator.ApplyGate(gate.qubits, gate.matrix.data(), state);
    } else {
      simulator.ApplyControlledGate(gate.qubits, gate.controlled_by,
                                    gate.cmask, gate.matrix.data(), state);
    }
  }
}

/**
 * Applies the given gate dagger to the simulator state. If the gate matrix is
 *   unitary then this is equivalent to applying the inverse gate. Ignores
 *   measurement gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Gate>
inline void ApplyGateDagger(const Simulator& simulator, const Gate& gate,
                            typename Simulator::State& state) {
  if (gate.kind != gate::kMeasurement) {
    auto matrix = gate.matrix;
    MatrixDagger(unsigned{1} << gate.qubits.size(), matrix);

    if (gate.controlled_by.size() == 0) {
      simulator.ApplyGate(gate.qubits, matrix.data(), state);
    } else {
      simulator.ApplyControlledGate(gate.qubits, gate.controlled_by,
                                    gate.cmask, matrix.data(), state);
    }
  }
}

/**
 * Applies the given gate to the simulator state.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @param mresults As an input parameter, this can be empty or this can
 *   contain the results of the previous measurements. If gate is a measurement
 *   gate then after a successful run, the measurement result will be added to
 *   this.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Gate, typename Rgen>
inline bool ApplyGate(
    const typename Simulator::StateSpace& state_space,
    const Simulator& simulator, const Gate& gate, Rgen& rgen,
    typename Simulator::State& state,
    std::vector<typename Simulator::StateSpace::MeasurementResult>& mresults) {
  if (gate.kind == gate::kMeasurement) {
    auto measure_result = state_space.Measure(gate.qubits, rgen, state);
    if (measure_result.valid) {
      mresults.push_back(std::move(measure_result));
    } else {
      return false;
    }
  } else {
    ApplyGate(simulator, gate, state);
  }

  return true;
}

/**
 * Applies the given gate to the simulator state, discarding measurement
 *   results.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Gate, typename Rgen>
inline bool ApplyGate(const typename Simulator::StateSpace& state_space,
                      const Simulator& simulator, const Gate& gate, Rgen& rgen,
                      typename Simulator::State& state) {
  using MeasurementResult = typename Simulator::StateSpace::MeasurementResult;
  std::vector<MeasurementResult> discarded_results;
  return
      ApplyGate(state_space, simulator, gate, rgen, state, discarded_results);
}

/**
 * Applies the given fused gate to the simulator state. Ignores measurement
 *   gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Gate>
inline void ApplyFusedGate(const Simulator& simulator, const Gate& gate,
                           typename Simulator::State& state) {
  if (gate.kind != gate::kMeasurement) {
    using fp_type = typename Simulator::fp_type;
    auto matrix = CalculateFusedMatrix<fp_type>(gate);
    if (gate.parent->controlled_by.size() == 0) {
      simulator.ApplyGate(gate.qubits, matrix.data(), state);
    } else {
      simulator.ApplyControlledGate(gate.qubits, gate.parent->controlled_by,
                                    gate.parent->cmask, matrix.data(), state);
    }
  }
}

/**
 * Applies the given fused gate dagger to the simulator state. If the gate
 *   matrix is unitary then this is equivalent to applying the inverse gate.
 *   Ignores measurement gates.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param state The state of the system, to be updated by this method.
 */
template <typename Simulator, typename Gate>
inline void ApplyFusedGateDagger(const Simulator& simulator, const Gate& gate,
                                 typename Simulator::State& state) {
  if (gate.kind != gate::kMeasurement) {
    using fp_type = typename Simulator::fp_type;
    auto matrix = CalculateFusedMatrix<fp_type>(gate);
    MatrixDagger(unsigned{1} << gate.qubits.size(), matrix);
    if (gate.parent->controlled_by.size() == 0) {
      simulator.ApplyGate(gate.qubits, matrix.data(), state);
    } else {
      simulator.ApplyControlledGate(gate.qubits, gate.parent->controlled_by,
                                    gate.parent->cmask, matrix.data(), state);
    }
  }
}

/**
 * Applies the given fused gate to the simulator state.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @param mresults As an input parameter, this can be empty or this can
 *   contain the results of the previous measurements. If gate is a measurement
 *   gate then after a successful run, the measurement result will be added to
 *   this.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Gate, typename Rgen>
inline bool ApplyFusedGate(
    const typename Simulator::StateSpace& state_space,
    const Simulator& simulator, const Gate& gate, Rgen& rgen,
    typename Simulator::State& state,
    std::vector<typename Simulator::StateSpace::MeasurementResult>& mresults) {
  if (gate.kind == gate::kMeasurement) {
    auto measure_result = state_space.Measure(gate.qubits, rgen, state);
    if (measure_result.valid) {
      mresults.push_back(std::move(measure_result));
    } else {
      return false;
    }
  } else {
    ApplyFusedGate(simulator, gate, state);
  }

  return true;
}

/**
 * Applies the given fused gate to the simulator state, discarding measurement
 *   results.
 * @param state_space StateSpace object required to perform measurements.
 * @param simulator Simulator object. Provides specific implementations for
 *   applying gates.
 * @param gate The gate to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state The state of the system, to be updated by this method.
 * @return True if the measurement performed successfully; false otherwise.
 */
template <typename Simulator, typename Gate, typename Rgen>
inline bool ApplyFusedGate(const typename Simulator::StateSpace& state_space,
                           const Simulator& simulator, const Gate& gate,
                           Rgen& rgen, typename Simulator::State& state) {
  using MeasurementResult = typename Simulator::StateSpace::MeasurementResult;
  std::vector<MeasurementResult> discarded_results;
  return ApplyFusedGate(
      state_space, simulator, gate, rgen, state, discarded_results);
}

}  // namespace qsim

#endif  // GATE_APPL_H_
