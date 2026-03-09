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

#include <stdexcept>

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
    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    explicit Factory(const py::dict& options) {
      verbosity = ParseOptions<unsigned>(options, "v\0");
      nwt = ParseOptions<unsigned>(options, "gnwt\0");

      if (!mp.Initialized()) {
        using MP = qsim::MultiProcessCuStateVecEx;

        if (!mp.ValidNetworkType(nwt)) {
          throw std::invalid_argument("Invalid network type.");
        }

        unsigned l = ParseOptions<unsigned>(options, "glbuf\0");
        uint64_t buffer_size = uint64_t{1} << l;

        MP::NetworkType network_type = static_cast<MP::NetworkType>(nwt);

        MP::Parameter param;
        param.transfer_buffer_size = buffer_size;
        param.network_type = network_type;

        mp.Initialize(param);

        if (verbosity > 2 && mp.Initialized()) {
          qsim::IO::messagef("transfer_buf_size=%lu\n", buffer_size);
        }
      }

      if (!mp.Initialized()) {
        if (!StateSpace::ValidDeviceNetworkType(nwt)) {
          throw std::invalid_argument("Invalid device network type.");
        }
      }
    }

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
      using NetworkType = StateSpace::DeviceNetworkType;

      StateSpace::Parameter param;
      param.device_network_type = static_cast<NetworkType>(nwt);
      param.verbosity = verbosity;

      return StateSpace{mp, param};
    }

    Simulator CreateSimulator() const {
      return Simulator{};
    }

    unsigned verbosity = 0;
    unsigned nwt = 0;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}

#include "../pybind_main.cpp"
