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

#include <custatevecEx.h>

#include "pybind_main_custatevecex.h"

#include "../../lib/fuser_mqubit.h"
#include "../../lib/gates_cirq.h"
#include "../../lib/io.h"
#include "../../lib/multiprocess_custatevecex.h"
#include "../../lib/run_custatevecex.h"
#include "../../lib/simulator_custatevecex.h"

namespace {

qsim::MultiProcessCuStateVecEx mp;

}  // namespace {

namespace qsim {
  using Simulator = SimulatorCuStateVecEx<float>;

  struct Factory {
    // num_sim_threads, num_state_threads and num_dblocks are unused, but kept
    // for consistency with other factories.
    Factory(unsigned num_sim_threads,
            unsigned num_state_threads,
            unsigned num_dblocks) {
      if (!mp.initialized()) {
        mp.initialize();
      }
    }

    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    using Gate = Cirq::GateCirq<float>;
    using Runner = CuStateVecExRunner<IO, Factory>;
    struct RunnerParameter : public Runner::Parameter {
      // max_fused_size is not used, but kept for consistency.
      unsigned max_fused_size = 2;
    };
    using NoisyRunner = qsim::QuantumTrajectorySimulator<IO, Gate, Runner>;
    struct NoisyRunnerParameter : public NoisyRunner::Parameter {
      // max_fused_size is not used, but kept for consistency.
      unsigned max_fused_size = 2;
    };

    StateSpace CreateStateSpace() const {
      return StateSpace{mp};
    }

    Simulator CreateSimulator() const {
      return Simulator{};
    }
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}

#include "../pybind_main.cpp"
