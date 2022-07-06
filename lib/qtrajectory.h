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
#include <complex>
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
    /**
     * If false, do not apply deferred operators after the main loop for
     * the "primary" noise trajectory, that is the trajectory in which
     * the primary (the first operators in their respective channels) Kraus
     * operators are sampled for each channel and there are no measurements
     * in the computational basis. This can be used to speed up simulations
     * of circuits with weak noise and without measurements by reusing
     * the primary trajectory results. There is an additional condition for
     * RunBatch. In this case, the deferred operators after the main loop are
     * still applied for the first occurence of the primary trajectory.
     * The primary Kraus operators should have the highest sampling
     * probabilities to achieve the highest speedup.
     *
     * It is the client's responsibility to collect the primary trajectory
     * results and to reuse them.
     */
    bool apply_last_deferred_ops = true;
  };

  /**
   * Struct with statistics to populate by RunBatch and RunOnce methods.
   */
  struct Stat {
    /**
     * Indices of sampled Kraus operator indices and/or measured bitstrings.
     */
    std::vector<uint64_t> samples;
    /**
     * True if the "primary" noise trajectory is sampled, false otherwise.
     */
    bool primary;
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
   *   measured bitstrings (const Stat&)] and any number of optional parameters.
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
   *   measured bitstrings (const Stat&)] and any number of optional parameters.
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

    Stat stat;
    bool had_primary_realization = false;

    for (uint64_t r = r0; r < r1; ++r) {
      if (!state_space.IsNull(state)) {
        state_space.SetStateZero(state);
      }

      bool apply_last_deferred_ops =
          param.apply_last_deferred_ops || !had_primary_realization;

      if (!RunIteration(param, apply_last_deferred_ops, num_qubits, cbeg, cend,
                        r, state_space, simulator, gates, state, stat)) {
        return false;
      }

      if (stat.primary && !had_primary_realization) {
        had_primary_realization = true;
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
   * @param state The state of the system, to be updated by this method.
   * @param stat Statistics of sampled Kraus operator indices and/or measured
   *   bitstrings, to be populated by this method.
   * @return True if the simulation completed successfully; false otherwise.
   */
  static bool RunOnce(const Parameter& param,
                      const NoisyCircuit<Gate>& circuit, uint64_t r,
                      const StateSpace& state_space, const Simulator& simulator,
                      State& state, Stat& stat) {
    return RunOnce(param, circuit.num_qubits, circuit.channels.begin(),
                   circuit.channels.end(), r, state_space, simulator,
                   state, stat);
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
   * @param state The state of the system, to be updated by this method.
   * @param stat Statistics of sampled Kraus operator indices and/or measured
   *   bitstrings, to be populated by this method.
   * @return True if the simulation completed successfully; false otherwise.
   */
  static bool RunOnce(const Parameter& param, unsigned num_qubits,
                      ncircuit_iterator<Gate> cbeg,
                      ncircuit_iterator<Gate> cend,
                      uint64_t r, const StateSpace& state_space,
                      const Simulator& simulator, State& state, Stat& stat) {
    std::vector<const Gate*> gates;
    gates.reserve(4 * std::size_t(cend - cbeg));

    if (!RunIteration(param, param.apply_last_deferred_ops, num_qubits, cbeg,
                      cend, r, state_space, simulator, gates, state, stat)) {
      return false;
    }

    return true;
  }

 private:
  static bool RunIteration(const Parameter& param,
                           bool apply_last_deferred_ops, unsigned num_qubits,
                           ncircuit_iterator<Gate> cbeg,
                           ncircuit_iterator<Gate> cend,
                           uint64_t rep, const StateSpace& state_space,
                           const Simulator& simulator,
                           std::vector<const Gate*>& gates,
                           State& state, Stat& stat) {
    if (param.collect_kop_stat || param.collect_mea_stat) {
      stat.samples.reserve(std::size_t(cend - cbeg));
      stat.samples.resize(0);
    }

    if (state_space.IsNull(state)) {
      state = CreateState(num_qubits, state_space);
      if (state_space.IsNull(state)) {
        return false;
      }

      state_space.SetStateZero(state);
    }

    gates.resize(0);

    RGen rgen(rep);
    std::uniform_real_distribution<double> distr(0.0, 1.0);

    bool unitary = true;
    stat.primary = true;

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

        stat.primary = false;

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

      double max_prob = 0;
      std::size_t max_prob_index = 0;

      // Perform sampling of Kraus operators using norms of updated states.
      for (std::size_t i = 0; i < channel.size(); ++i) {
        const auto& kop = channel[i];

        if (kop.unitary) continue;

        double prob = std::real(
            simulator.ExpectationValue(kop.qubits, kop.kd_k.data(), state));

        if (prob > max_prob) {
          max_prob = prob;
          max_prob_index = i;
        }

        cp += prob - kop.prob;

        if (r < cp || i == channel.size() - 1) {
          // Sample ith Kraus operator if r < cp
          // Sample the highest probability Kraus operator if r is greater
          // than the sum of all probablities due to round-off errors.
          uint64_t k = r < cp ? i : max_prob_index;

          DeferOps(channel[k].ops, gates);
          CollectStat(param.collect_kop_stat, k, stat);

          unitary = false;

          break;
        }
      }
    }

    if (apply_last_deferred_ops || !stat.primary) {
      if (!ApplyDeferredOps(param, num_qubits, simulator, gates, state)) {
        return false;
      }

      NormalizeState(!unitary, state_space, unitary, state);
    }

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

  static void CollectStat(bool collect_stat, uint64_t i, Stat& stat) {
    if (collect_stat) {
      stat.samples.push_back(i);
    }

    if (i != 0) {
      stat.primary = false;
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
