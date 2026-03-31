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

#include "pybind_main_hip.h"

#include "../../lib/fuser_mqubit.h"
#include "../../lib/gates_cirq.h"
#include "../../lib/io.h"
#include "../../lib/run_qsim.h"
#include "../../lib/simulator_cuda.h"

namespace qsim {
  using Simulator = SimulatorCUDA<float>;

  struct Factory {
     explicit Factory(const py::dict& options) {
      ss_params.num_threads = ParseOptions<unsigned>(options, "gsst\0");
      ss_params.num_dblocks = ParseOptions<unsigned>(options, "gdb\0");
    }

    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    using Gate = Cirq::GateCirq<float>;
    using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Gate>, Factory>;
    using RunnerQT =
        QSimRunner<IO, MultiQubitGateFuser<IO, const Gate*>, Factory>;
    using RunnerParameter = Runner::Parameter;
    using NoisyRunner = qsim::QuantumTrajectorySimulator<IO, Gate, RunnerQT>;
    using NoisyRunnerParameter = NoisyRunner::Parameter;

    StateSpace CreateStateSpace() const {
      return StateSpace(ss_params);
    }

    Simulator CreateSimulator() const {
      return Simulator();
    }

    StateSpace::Parameter ss_params;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}

#include "../pybind_main.cpp"
