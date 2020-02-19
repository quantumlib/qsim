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

#include <string>
#include <vector>

#include "gates_appl.h"
#include "util.h"

namespace qsim {

// Helper struct to run qsim.

template <typename IO, typename Fuser, typename Simulator>
struct QSimRunner final {
  struct Parameter {
    unsigned num_threads;
    unsigned verbosity;
  };

  template <typename Circuit, typename MeasurementFunc>
  static bool Run(const Parameter& param, unsigned maxtime,
                  const Circuit& circuit, MeasurementFunc measure,
                  typename Simulator::fp_type* ext_state = nullptr) {
    std::vector<unsigned> times_to_measure_at{maxtime};
    return Run(param, times_to_measure_at, circuit, measure, ext_state);
  }

  template <typename Circuit, typename MeasurementFunc>
  static bool Run(const Parameter& param,
                  const std::vector<unsigned>& times_to_measure_at,
                  const Circuit& circuit, MeasurementFunc measure,
                  typename Simulator::fp_type* ext_state = nullptr) {
    double t0, t1;
    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    using StateSpace = typename Simulator::StateSpace;
    StateSpace state_space(circuit.num_qubits, param.num_threads);

    auto state = ext_state == nullptr ? state_space.CreateState() :
                                        state_space.CreateState(ext_state);

    if (state_space.IsNull(state)) {
      IO::errorf("not enough memory: is the number of qubits too large?\n");
      return false;
    }

    state_space.SetStateZero(state);
    Simulator simulator(circuit.num_qubits, param.num_threads);

    auto fused_gates = Fuser::FuseGates(circuit.num_qubits, circuit.gates,
                                        times_to_measure_at);

    unsigned cur_time_index = 0;

    // Apply fused gates.
    for (std::size_t i = 0; i < fused_gates.size(); ++i) {
      if (param.verbosity > 1) {
        t1 = GetTime();
      }

      ApplyFusedGate(simulator, fused_gates[i], state);

      if (param.verbosity > 1) {
        double t2 = GetTime();
        IO::messagef("gate %lu done in %g seconds\n", i, t2 - t1);
      }

      unsigned t = times_to_measure_at[cur_time_index];

      if (i == fused_gates.size() - 1 || t < fused_gates[i + 1].time) {
        // Call back to perform measurements.
        measure(cur_time_index, state_space, state);
        ++cur_time_index;
      }
    }

    if (param.verbosity > 0) {
      double t2 = GetTime();
      IO::messagef("time elapsed %g seconds.\n", t2 - t0);
    }

    return true;
  }
};

}  // namespace qsim

#endif  // RUN_QSIM_H_
