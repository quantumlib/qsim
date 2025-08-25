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

#include "pybind_main_sse.h"

#include "../../lib/formux.h"
#include "../../lib/fuser_mqubit.h"
#include "../../lib/gates_cirq.h"
#include "../../lib/io.h"
#include "../../lib/run_qsim.h"
#include "../../lib/simulator_sse.h"
#include "../../lib/util_cpu.h"

namespace qsim {
  template <typename For>
  using Simulator = SimulatorSSE<For>;

  struct Factory {
    // num_state_threads and num_dblocks are unused, but kept for consistency
    // with the GPU Factory.
    Factory(
      unsigned num_sim_threads,
      unsigned num_state_threads,
      unsigned num_dblocks) : num_threads(num_sim_threads) {}

    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;

    using Gate = Cirq::GateCirq<float>;
    using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Gate>, Factory>;
    using RunnerQT =
        QSimRunner<IO, MultiQubitGateFuser<IO, const Gate*>, Factory>;
    using RunnerParameter = Runner::Parameter;
    using NoisyRunner = qsim::QuantumTrajectorySimulator<IO, Gate, RunnerQT>;
    using NoisyRunnerParameter = NoisyRunner::Parameter;

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    Simulator CreateSimulator() const {
      return Simulator(num_threads);
    }

    unsigned num_threads;
  };
}

#include "../pybind_main.cpp"
