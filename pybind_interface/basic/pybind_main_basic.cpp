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

#include "pybind_main_basic.h"

#include "../../lib/formux.h"
#include "../../lib/simulator_basic.h"
#include "../../lib/util_cpu.h"

namespace qsim {
  template <typename For>
  using Simulator = SimulatorBasic<For>;

  struct Factory {
    // num_state_threads and num_dblocks are unused, but kept for consistency
    // with the GPU Factory.
    Factory(
      unsigned num_sim_threads,
      unsigned num_state_threads,
      unsigned num_dblocks) : num_threads(num_sim_threads) {}

    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;

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
