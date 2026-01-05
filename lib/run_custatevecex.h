// Copyright 2025 Google LLC. All Rights Reserved.
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

#ifndef RUN_CUSTATEVECEX_H_
#define RUN_CUSTATEVECEX_H_

#include <random>
#include <string>
#include <vector>

#include <custatevecEx.h>

#include "circuit.h"
#include "util.h"
#include "util_custatevec.h"
#include "util_custatevecex.h"

namespace qsim {

/**
 * Helper struct for running qsim with the cuStateVecEx library.
 */
template <typename IO, typename Factory, typename RGen = std::mt19937>
struct CuStateVecExRunner final {
 public:
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using MeasurementResult = typename StateSpace::MeasurementResult;

  /**
   * User-specified parameters for simulation.
   */
  struct Parameter {
    /**
     * Random number generator seed to apply measurement gates.
     */
    uint64_t seed;

    unsigned verbosity = 0;
  };

  /**
   * Runs the given circuit, only measuring at the end.
   * @param param Options for gate fusion, parallelism and logging.
   * @param factory Object to create simulators and state spaces.
   * @param circuit The circuit to be simulated.
   * @param measure Function that performs measurements (in the sense of
   *   computing expectation values, etc).
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit, typename MeasurementFunc>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, MeasurementFunc measure) {
    return Run(param, factory, {circuit.gates.back().time}, circuit, measure);
  }

  /**
   * Runs the given circuit, measuring at user-specified times.
   * @param param Options for gate fusion, parallelism and logging.
   * @param factory Object to create simulators and state spaces.
   * @param times_to_measure_at Time steps at which to perform measurements.
   * @param circuit The circuit to be simulated.
   * @param measure Function that performs measurements (in the sense of
   *   computing expectation values, etc).
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit, typename MeasurementFunc>
  static bool Run(const Parameter& param, const Factory& factory,
                  const std::vector<unsigned>& times_to_measure_at,
                  const Circuit& circuit, MeasurementFunc measure) {
    std::vector<MeasurementResult> discarded_results;

    StateSpace state_space = factory.CreateStateSpace();
    Simulator simulator = factory.CreateSimulator();

    auto state = state_space.Create(circuit.num_qubits);
    if (state_space.IsNull(state)) {
      IO::errorf("not enough memory: is the number of qubits too large?\n");
      return false;
    }

    state_space.SetStateZero(state);

    return Run(param, circuit, state_space, simulator, state,
               times_to_measure_at, measure, discarded_results);
  }

  /**
   * Runs the given circuit and make the final state available to the caller,
   * recording the result of any intermediate measurements in the circuit.
   * @param param Options for gate fusion, parallelism and logging.
   * @param factory Object to create simulators and state spaces.
   * @param circuit The circuit to be simulated.
   * @param state As an input parameter, this should contain the initial state
   *   of the system. After a successful run, it will be populated with the
   *   final state of the system.
   * @param measure_results As an input parameter, this should be empty.
   *   After a successful run, this will contain all measurements results from
   *   the run, ordered by time and qubit index.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, State& state,
                  std::vector<MeasurementResult>& measure_results) {
    auto measure = [](unsigned, const StateSpace&, const State&) {};

    StateSpace state_space = factory.CreateStateSpace();
    Simulator simulator = factory.CreateSimulator();

    return Run(param, circuit, state_space, simulator, state,
               {}, measure, measure_results);
  }

  /**
   * Runs the given circuit and make the final state available to the caller,
   * discarding the result of any intermediate measurements in the circuit.
   * @param param Options for gate fusion, parallelism and logging.
   * @param factory Object to create simulators and state spaces.
   * @param circuit The circuit to be simulated.
   * @param state As an input parameter, this should contain the initial state
   *   of the system. After a successful run, it will be populated with the
   *   final state of the system.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, State& state) {
    auto measure = [](unsigned, const StateSpace&, const State&) {};

    StateSpace state_space = factory.CreateStateSpace();
    Simulator simulator = factory.CreateSimulator();

    std::vector<MeasurementResult> discarded_results;

    return Run(param, circuit, state_space, simulator, state,
               {}, measure, discarded_results);
  }

  /**
   * Runs the given circuit and make the final state available to the caller,
   * recording the result of any intermediate measurements in the circuit.
   * @param param Options for gate fusion, parallelism and logging.
   * @param circuit The circuit to be simulated.
   * @param state_space StateSpace object required to perform measurements.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param state As an input parameter, this should contain the initial state
   *   of the system. After a successful run, it will be populated with the
   *   final state of the system.
   * @param measure_results As an input parameter, this should be empty.
   *   After a successful run, this will contain all measurements results from
   *   the run, ordered by time and qubit index.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  std::vector<MeasurementResult>& measure_results) {
    auto measure = [](unsigned, const StateSpace&, const State&) {};

    return Run(param, circuit, state_space, simulator, state,
               {}, measure, measure_results);
  }

  /**
   * Runs the given circuit and make the final state available to the caller,
   * discarding the result of any intermediate measurements in the circuit.
   * @param param Options for gate fusion, parallelism and logging.
   * @param circuit The circuit to be simulated.
   * @param state_space StateSpace object required to perform measurements.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param state As an input parameter, this should contain the initial state
   *   of the system. After a successful run, it will be populated with the
   *   final state of the system.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state) {
    auto measure = [](unsigned, const StateSpace&, const State&) {};

    std::vector<MeasurementResult> discarded_results;

    return Run(param, circuit, state_space, simulator, state,
               {}, measure, discarded_results);
  }

 private:
  template <typename Circuit, typename MeasurementFunc>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  const std::vector<unsigned>& times_to_measure_at,
                  MeasurementFunc measure,
                  std::vector<MeasurementResult>& measure_results) {
    double t0 = 0.0;

    RGen rgen(param.seed);

    custatevecExSVUpdaterDescriptor_t sv_updater = nullptr;
    custatevecExDictionaryDescriptor_t sv_updater_config = nullptr;

    ErrorCheck(custatevecExConfigureSVUpdater(
        &sv_updater_config, StateSpace::kStateDataType, nullptr, 0));

    ErrorCheck(
        custatevecExSVUpdaterCreate(&sv_updater, sv_updater_config, nullptr));
    ErrorCheck(custatevecExDictionaryDestroy(sv_updater_config));

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    unsigned cur_time_index = 0;

    using Gates = detail::Gates<Circuit>;
    const auto& gates = Gates::get(circuit);

    for (std::size_t i = 0; i < gates.size(); ++i) {
      const auto& gate = Gates::gate(gates[i]);
      unsigned num_qubits = gate.qubits.size();
      unsigned num_cqubits = gate.controlled_by.size();

      if (gate.kind == gate::kMeasurement) {
        ErrorCheck(
            custatevecExSVUpdaterApply(sv_updater, state.get(), nullptr, 0));
        ErrorCheck(custatevecExSVUpdaterClear(sv_updater));

        auto measure_result = state_space.Measure(gate.qubits, rgen, state);
        if (measure_result.valid) {
          measure_results.push_back(std::move(measure_result));
        } else {
          IO::errorf("measurement failed.\n");
          return false;
        }
      } else if (num_cqubits == 0) {
        if (num_qubits == 0) {
          ErrorCheck(
            custatevecExSVUpdaterApply(sv_updater, state.get(), nullptr, 0));
          ErrorCheck(custatevecExSVUpdaterClear(sv_updater));

          simulator.ApplyGate(gate.qubits, gate.matrix.data(), state);
        } else {
          ErrorCheck(custatevecExSVUpdaterEnqueueMatrix(
              sv_updater, gate.matrix.data(), StateSpace::kMatrixDataType,
              StateSpace::kExMatrixType, StateSpace::kMatrixLayout, 0,
              reinterpret_cast<const int32_t*>(gate.qubits.data()),
              num_qubits, nullptr, nullptr, 0));
        }
      } else {
        std::vector<int32_t> control_bits;
        control_bits.reserve(num_cqubits);

        for (std::size_t i = 0; i < num_cqubits; ++i) {
          control_bits.push_back((gate.cmask >> i) & 1);
        }

        ErrorCheck(custatevecExSVUpdaterEnqueueMatrix(
            sv_updater, gate.matrix.data(), StateSpace::kMatrixDataType,
            StateSpace::kExMatrixType, StateSpace::kMatrixLayout, 0,
            reinterpret_cast<const int32_t*>(gate.qubits.data()), num_qubits,
            reinterpret_cast<const int32_t*>(gate.controlled_by.data()),
            control_bits.data(), num_cqubits));
      }

      if (times_to_measure_at.size() > 0) {
        unsigned t = times_to_measure_at[cur_time_index];

        if (i == gates.size() - 1 || t < Gates::gate(gates[i + 1]).time) {
          ErrorCheck(
              custatevecExSVUpdaterApply(sv_updater, state.get(), nullptr, 0));
          ErrorCheck(custatevecExSVUpdaterClear(sv_updater));

          // Call back to perform measurements.
          measure(cur_time_index, state_space, state);
          ++cur_time_index;
        }
      }
    }

    ErrorCheck(custatevecExSVUpdaterApply(sv_updater, state.get(), nullptr, 0));

    if (param.verbosity > 0) {
      state_space.DeviceSync();
      double t1 = GetTime();
      IO::messagef("simu time is %g seconds.\n", t1 - t0);
    }

    ErrorCheck(custatevecExSVUpdaterDestroy(sv_updater));

    return true;
  }
};

}  // namespace qsim

#endif  // RUN_CUSTATEVECEX_H_
