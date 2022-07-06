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

#include "pybind_main_cuda.h"

#include "../../lib/simulator_cuda.h"

namespace qsim {
  using Simulator = SimulatorCUDA<float>;

  struct Factory {
    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    Factory(
      unsigned num_sim_threads,
      unsigned num_state_threads,
      unsigned num_dblocks
    ) : ss_params{num_state_threads, num_dblocks},
        sim_params{num_sim_threads} {}

    StateSpace CreateStateSpace() const {
      return StateSpace(ss_params);
    }

    Simulator CreateSimulator() const {
      return Simulator(sim_params);
    }

    StateSpace::Parameter ss_params;
    Simulator::Parameter sim_params;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}

#include "../pybind_main.cpp"
