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

#ifndef QTRAJECTORY_H_
#define QTRAJECTORY_H_

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "circuit_noisy.h"
#include "gate.h"
#include "gate_appl.h"

namespace qsim {

/**
 * Quantum trajectory simulator.
 */
template <typename IO, typename Gate,
          template <typename, typename> class FuserT, typename Simulator,
          typename RGen = std::mt19937>
class QuantumTrajectorySimulator {
 public:
  using Fuser = FuserT<IO, const Gate*>;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename Simulator::State;
  using MeasurementResult = typename StateSpace::MeasurementResult;

  /**
   * User-specified parameters for the simulator.
   */
  struct Parameter : public Fuser::Parameter {
    /**
     * If true, collect statistics of sampled Kraus operator indices.
     */
    bool collect_kop_stat = false;
    /**
     * If true, collect statistics of measured bitstrings.
     */
    bool collect_mea_stat = false;
    /**
     * If true, normalize the state vector before performing measurements.
     */
    bool normalize_before_mea_gates = true;
  };

  /**
   * Runs the given noisy circuit performing repetitions. Each repetition is
   * seeded by repetition ID.
   * @param param Options for the quantum trajectory simulator.
   * @param circuit The noisy circuit to be simulated.
   * @param r0, r1 The range of repetition IDs [r0, r1) to perform repetitions.
   * @param state_space StateSpace object required to manipulate state vector.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param measure Function that performs measurements (in the sense of
   *   computing expectation values, etc). This function should have three
   *   required parameters [repetition ID (uint64_t), final state vector
   *   (const State&), statistics of sampled Kraus operator indices and/or
   *   measured bitstrings (const std::vector<uint64_t>&)] and any number of
   *   optional parameters.
   * @param args Optional arguments for the 'measure' function.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename MeasurementFunc, typename... Args>
  static bool RunBatch(const Parameter& param,
                       const NoisyCircuit<Gate>& circuit,
                       uint64_t r0, uint64_t r1, const StateSpace& state_space,
                       const Simulator& simulator, MeasurementFunc&& measure,
                       Args&&... args) {
    return RunBatch(param, circuit.num_qubits, circuit.channels.begin(),
                    circuit.channels.end(), r0, r1, state_space, simulator,
                    measure, args...);
  }

  /**
   * Runs the given noisy circuit performing repetitions. Each repetition is
   * seeded by repetition ID.
   * @param param Options for the quantum trajectory simulator.
   * @param num_qubits The number of qubits acted on by the circuit.
   * @param cbeg, cend The range of channels [cbeg, cend) to run the circuit.
   * @param r0, r1 The range of repetition IDs [r0, r1) to perform repetitions.
   * @param state_space StateSpace object required to manipulate state vector.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param measure Function that performs measurements (in the sense of
   *   computing expectation values, etc). This function should have three
   *   required parameters [repetition ID (uint64_t), final state vector
   *   (const State&), statistics of sampled Kraus operator indices and/or
   *   measured bitstrings (const std::vector<uint64_t>&)] and any number of
   *   optional parameters.
   * @param args Optional arguments for the 'measure' function.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename MeasurementFunc, typename... Args>
  static bool RunBatch(const Parameter& param, unsigned num_qubits,
                       ncircuit_iterator<Gate> cbeg,
                       ncircuit_iterator<Gate> cend,
                       uint64_t r0, uint64_t r1, const StateSpace& state_space,
                       const Simulator& simulator, MeasurementFunc&& measure,
                       Args&&... args) {
    std::vector<const Gate*> gates;
    gates.reserve(4 * std::size_t(cend - cbeg));

    State state = state_space.Null();
    State scratch = state_space.Null();

    std::vector<uint64_t> stat;

    for (uint64_t r = r0; r < r1; ++r) {
      if (!state_space.IsNull(state)) {
        state_space.SetStateZero(state);
      }

      if (!RunIteration(param, num_qubits, cbeg, cend, r,
                        state_space, simulator, gates, scratch, state, stat)) {
        return false;
      }

      measure(r, state, stat, args...);
    }

    return true;
  }

  /**
   * Runs the given noisy circuit one time.
   * @param param Options for the quantum trajectory simulator.
   * @param circuit The noisy circuit to be simulated.
   * @param r The repetition ID. The random number generator is seeded by 'r'.
   * @param state_space StateSpace object required to manipulate state vector.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param scratch A temporary state vector. Used for samping Kraus operators.
   * @param state The state of the system, to be updated by this method.
   * @param stat Statistics of sampled Kraus operator indices and/or measured
   *   bitstrings, to be populated by this method.
   * @return True if the simulation completed successfully; false otherwise.
   */
  static bool RunOnce(const Parameter& param,
                      const NoisyCircuit<Gate>& circuit, uint64_t r,
                      const StateSpace& state_space, const Simulator& simulator,
                      State& scratch, State& state,
                      std::vector<uint64_t>& stat) {
    return RunOnce(param, circuit.num_qubits, circuit.channels.begin(),
                   circuit.channels.end(), r, state_space, simulator,
                   scratch, state, stat);
  }

  /**
   * Runs the given noisy circuit one time.
   * @param param Options for the quantum trajectory simulator.
   * @param num_qubits The number of qubits acted on by the circuit.
   * @param cbeg, cend The range of channels [cbeg, cend) to run the circuit.
   * @param circuit The noisy circuit to be simulated.
   * @param r The repetition ID. The random number generator is seeded by 'r'.
   * @param state_space StateSpace object required to manipulate state vector.
   * @param simulator Simulator object. Provides specific implementations for
   *   applying gates.
   * @param scratch A temporary state vector. Used for samping Kraus operators.
   * @param state The state of the system, to be updated by this method.
   * @param stat Statistics of sampled Kraus operator indices and/or measured
   *   bitstrings, to be populated by this method.
   * @return True if the simulation completed successfully; false otherwise.
   */
  static bool RunOnce(const Parameter& param, unsigned num_qubits,
                      ncircuit_iterator<Gate> cbeg,
                      ncircuit_iterator<Gate> cend,
                      uint64_t r, const StateSpace& state_space,
                      const Simulator& simulator, State& scratch, State& state,
                      std::vector<uint64_t>& stat) {
    std::vector<const Gate*> gates;
    gates.reserve(4 * std::size_t(cend - cbeg));

    if (!RunIteration(param, num_qubits, cbeg, cend, r,
                      state_space, simulator, gates, scratch, state, stat)) {
      return false;
    }

    return true;
  }

 private:
  static bool RunIteration(const Parameter& param, unsigned num_qubits,
                           ncircuit_iterator<Gate> cbeg,
                           ncircuit_iterator<Gate> cend,
                           uint64_t rep, const StateSpace& state_space,
                           const Simulator& simulator,
                           std::vector<const Gate*>& gates, State& scratch,
                           State& state, std::vector<uint64_t>& stat) {
    if (param.collect_kop_stat || param.collect_mea_stat) {
      stat.reserve(std::size_t(cend - cbeg));
      stat.resize(0);
    }

    if (state_space.IsNull(state)) {
      state = CreateState(num_qubits, state_space);
      if (state_space.IsNull(state)) {
        return false;
      }

      state_space.SetStateZero(state);
    }

    gates.resize(0);
    stat.resize(0);

    RGen rgen(rep);
    std::uniform_real_distribution<double> distr(0.0, 1.0);

    bool unitary = true;

    for (auto it = cbeg; it != cend; ++it) {
      const auto& channel = *it;

      if (channel.size() == 0) continue;

      if (channel[0].kind == gate::kMeasurement) {
        // Measurement channel.

        if (!ApplyDeferredOps(param, num_qubits, simulator, gates, state)) {
          return false;
        }

        bool normalize = !unitary && param.normalize_before_mea_gates;
        NormalizeState(normalize, state_space, unitary, state);

        auto mresult = ApplyMeasurementGate(state_space, channel[0].ops[0],
                                            rgen, state);

        if (!mresult.valid) {
          return false;
        }

        CollectStat(param.collect_mea_stat, mresult.bits, stat);

        continue;
      }

      // "Normal" channel.

      double r = distr(rgen);
      double cp = 0;

      // Perform sampling of Kraus operators using probability bounds.
      for (std::size_t i = 0; i < channel.size(); ++i) {
        const auto& kop = channel[i];

        cp += kop.prob;

        if (r < cp) {
          DeferOps(kop.ops, gates);
          CollectStat(param.collect_kop_stat, i, stat);

          unitary = unitary && kop.unitary;

          break;
        }
      }

      if (r < cp) continue;

      if (!ApplyDeferredOps(param, num_qubits, simulator, gates, state)) {
        return false;
      }

      NormalizeState(!unitary, state_space, unitary, state);

      if (state_space.IsNull(scratch)) {
        scratch = CreateState(num_qubits, state_space);
        if (state_space.IsNull(scratch)) {
          return false;
        }
      }

      state_space.Copy(state, scratch);

      // Perform sampling of Kraus operators using norms of updated states.
      for (std::size_t i = 0; i < channel.size(); ++i) {
        const auto& kop = channel[i];

        if (kop.unitary) continue;

        // Apply the Kraus operator.
        if (kop.ops.size() == 1) {
          ApplyGate(simulator, kop.ops[0], state);
        } else {
          DeferOps(kop.ops, gates);

          if (!ApplyDeferredOps(param, num_qubits, simulator, gates, state)) {
            return false;
          }
        }

        double n2 = state_space.Norm(state);

        cp += n2 - kop.prob;

        if (r < cp || i == channel.size() - 1) {
          // Sample ith Kraus operator if r < cp
          // Sample the first Kraus operator if r is greater than the sum of
          // all probablities due to round-off errors.
          uint64_t k = r < cp ? i : 0;

          CollectStat(param.collect_kop_stat, k, stat);

          unitary = false;

          break;
        }

        state_space.Copy(scratch, state);
      }
    }

    if (!ApplyDeferredOps(param, num_qubits, simulator, gates, state)) {
      return false;
    }

    NormalizeState(!unitary, state_space, unitary, state);

    return true;
  }

  static State CreateState(unsigned num_qubits, const StateSpace& state_space) {
    auto state = state_space.Create(num_qubits);
    if (state_space.IsNull(state)) {
      IO::errorf("not enough memory: is the number of qubits too large?\n");
      return state_space.Null();
    }

    return state;
  }

  static bool ApplyDeferredOps(
      const Parameter& param, unsigned num_qubits, const Simulator& simulator,
      std::vector<const Gate*>& gates, State& state) {
    if (gates.size() > 0) {
      auto fgates = Fuser::FuseGates(param, num_qubits, gates);

      gates.resize(0);

      if (fgates.size() == 0) {
        return false;
      }

      for (const auto& fgate : fgates) {
        ApplyFusedGate(simulator, fgate, state);
      }
    }

    return true;
  }

  static MeasurementResult ApplyMeasurementGate(
      const StateSpace& state_space, const Gate& gate,
      RGen& rgen, State& state) {
    auto result = state_space.Measure(gate.qubits, rgen, state);

    if (!result.valid) {
      IO::errorf("measurement failed.\n");
    }

    return result;
  }

  static void DeferOps(
      const std::vector<Gate>& ops, std::vector<const Gate*>& gates) {
    for (const auto& op : ops) {
      gates.push_back(&op);
    }
  }

  static void CollectStat(bool collect_stat, uint64_t i,
                          std::vector<uint64_t>& stat) {
    if (collect_stat) {
      stat.push_back(i);
    }
  }

  static void NormalizeState(bool normalize, const StateSpace& state_space,
                             bool& flag, State& state) {
    if (normalize) {
      double a = 1.0 / std::sqrt(state_space.Norm(state));
      state_space.Multiply(a, state);
      flag = true;
    }
  }
};

}  // namespace qsim

#endif  // QTRAJECTORY_H_
