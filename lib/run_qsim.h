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

#ifndef RUN_QSIM_H_
#define RUN_QSIM_H_

#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "circuit.h"
#include "gate.h"
#include "gate_appl.h"
#include "operation_base.h"
#include "qubit_remap.h"
#include "util.h"

namespace qsim {

/**
 * Helper struct for running qsim.
 */
template <typename IO, typename Fuser, typename Factory,
          typename RGen = std::mt19937>
struct QSimRunner final {
 public:
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using MeasurementResult = typename StateSpace::MeasurementResult;

  template <typename Circuit>
  struct IsCircuit : std::false_type {};

  template <typename Gate>
  struct IsCircuit<qsim::Circuit<Gate>> : std::true_type {};

  /**
   * User-specified parameters for gate fusion and simulation.
   */
  struct Parameter : public Fuser::Parameter {
    /**
     * Random number generator seed to apply measurement gates.
     */
    uint64_t seed;
    /**
     * Remap logical qubits to improve state-vector cache locality.
     */
    bool cache_local_remap = false;
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
    unsigned time = OpTime(circuit.ops.back());
    return Run(param, factory, {time}, circuit, measure);
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
    double t0 = 0.0;
    double t1 = 0.0;

    if (param.verbosity > 1) {
      t0 = GetTime();
    }

    RGen rgen(param.seed);

    StateSpace state_space = factory.CreateStateSpace();

    auto state = state_space.Create(circuit.num_qubits);
    if (state_space.IsNull(state)) {
      IO::errorf("not enough memory: is the number of qubits too large?\n");
      return false;
    }

    state_space.SetStateZero(state);
    Simulator simulator = factory.CreateSimulator();

    if (param.verbosity > 1) {
      t1 = GetTime();
      IO::messagef("init time is %g seconds.\n", t1 - t0);
      t0 = GetTime();
    }

    const auto& ops = Operations<Circuit>::get(circuit);
    auto fused_ops = Fuser::FuseGates(param, circuit.num_qubits,
                                      ops, times_to_measure_at);

    if (fused_ops.size() == 0 && circuit.ops.size() > 0) {
      return false;
    }

    if (param.verbosity > 1) {
      t1 = GetTime();
      IO::messagef("fuse time is %g seconds.\n", t1 - t0);
    }

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    unsigned cur_time_index = 0;

    // Apply fused operations.
    for (std::size_t i = 0; i < fused_ops.size(); ++i) {
      if (param.verbosity > 3) {
        t1 = GetTime();
      }

      if (!ApplyGate(state_space, simulator, fused_ops[i], rgen, state)) {
        IO::errorf("measurement failed.\n");
        return false;
      }

      if (param.verbosity > 3) {
        state_space.DeviceSync();
        double t2 = GetTime();
        IO::messagef("gate %lu done in %g seconds.\n", i, t2 - t1);
      }

      unsigned t = times_to_measure_at[cur_time_index];

      if (i == fused_ops.size() - 1 || t < OpTime(fused_ops[i + 1])) {
        // Call back to perform measurements.
        measure(cur_time_index, state_space, state);
        ++cur_time_index;
      }
    }

    if (param.verbosity > 0) {
      state_space.DeviceSync();
      double t2 = GetTime();
      IO::messagef("time is %g seconds.\n", t2 - t0);
    }

    return true;
  }

  template <typename FusedOps>
  static bool RunFusedGates(
      const Parameter& param, const FusedOps& fused_ops,
      const StateSpace& state_space, const Simulator& simulator, State& state,
      std::vector<MeasurementResult>& measure_results) {
    double t0 = 0.0;
    double t1 = 0.0;

    RGen rgen(param.seed);
    measure_results.reserve(measure_results.size() + fused_ops.size());

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    // Apply fused operations.
    for (std::size_t i = 0; i < fused_ops.size(); ++i) {
      if (param.verbosity > 3) {
        t1 = GetTime();
      }

      if (!ApplyGate(state_space, simulator, fused_ops[i], rgen, state,
                     measure_results)) {
        IO::errorf("measurement failed.\n");
        return false;
      }

      if (param.verbosity > 3) {
        state_space.DeviceSync();
        double t2 = GetTime();
        IO::messagef("gate %lu done in %g seconds.\n", i, t2 - t1);
      }
    }

    if (param.verbosity > 0) {
      state_space.DeviceSync();
      double t2 = GetTime();
      IO::messagef("simu time is %g seconds.\n", t2 - t0);
    }

    return true;
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
    StateSpace state_space = factory.CreateStateSpace();
    Simulator simulator = factory.CreateSimulator();

    return Run(param, circuit, state_space, simulator, state, measure_results);
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
    StateSpace state_space = factory.CreateStateSpace();
    Simulator simulator = factory.CreateSimulator();

    std::vector<MeasurementResult> discarded_results;

    return Run(
        param, circuit, state_space, simulator, state, discarded_results);
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
  static bool Run(const Parameter& param, Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  std::vector<MeasurementResult>& measure_results) {
    if (param.cache_local_remap) {
      IO::errorf(
          "cache-local remap requires a logical_to_physical output map.\n");
      return false;
    }

    qubit_remap::QubitMap discarded_map;

    return Run(param, circuit, state_space, simulator, state,
               discarded_map, measure_results);
  }

  template <typename Circuit>
  static bool Run(const Parameter& param, const Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  std::vector<MeasurementResult>& measure_results) {
    auto circuit_copy = circuit;

    return Run(param, circuit_copy, state_space, simulator, state,
               measure_results);
  }

  template <typename Circuit>
  static bool Run(const Parameter& param, Circuit& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  qubit_remap::QubitMap& logical_to_physical,
                  std::vector<MeasurementResult>& measure_results) {
    using CircuitType = std::remove_cv_t<Circuit>;
    const auto& ops = Operations<CircuitType>::get(circuit);

    double t0 = 0.0;
    double t1 = 0.0;

    if (param.verbosity > 1) {
      t0 = GetTime();
    }

    if (param.verbosity > 1) {
      t1 = GetTime();
      IO::messagef("init time is %g seconds.\n", t1 - t0);
      t0 = GetTime();
    }

    auto fused_ops = Fuser::FuseGates(
        param, state.num_qubits(), ops, !param.cache_local_remap);

    if (fused_ops.size() == 0 && ops.size() > 0) {
      return false;
    }

    if (param.cache_local_remap) {
      if constexpr (IsCircuit<CircuitType>::value) {
        logical_to_physical =
            qubit_remap::RemapCircuit(fused_ops, circuit);
      } else {
        IO::errorf("cache-local remap requires a qsim::Circuit.\n");
        return false;
      }
    }

    if (param.verbosity > 1) {
      t1 = GetTime();
      IO::messagef("fuse time is %g seconds.\n", t1 - t0);
    }

    std::size_t previous_measurement_count = measure_results.size();
    if (!RunFusedGates(param, fused_ops, state_space, simulator, state,
                       measure_results)) {
      return false;
    }

    if (param.cache_local_remap) {
      for (std::size_t i = previous_measurement_count;
           i < measure_results.size(); ++i) {
        qubit_remap::detail::RemapMeasurementResult(
            logical_to_physical, measure_results[i]);
      }
    }

    return true;
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
    std::vector<MeasurementResult> discarded_results;

    return Run(
        param, circuit, state_space, simulator, state, discarded_results);
  }

  template <typename Gate>
  static bool Run(const Parameter& param, Circuit<Gate>& circuit,
                  const StateSpace& state_space, const Simulator& simulator,
                  State& state,
                  qubit_remap::QubitMap& logical_to_physical) {
    std::vector<MeasurementResult> discarded_results;

    return Run(param, circuit, state_space, simulator, state,
               logical_to_physical, discarded_results);
  }
};

}  // namespace qsim

#endif  // RUN_QSIM_H_
