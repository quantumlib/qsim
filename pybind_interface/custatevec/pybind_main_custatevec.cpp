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

#include <cublas_v2.h>
#include <custatevec.h>

#include "pybind_main_custatevec.h"

#include "../../lib/simulator_custatevec.h"

namespace qsim {

using Simulator = SimulatorCuStateVec<float>;

struct Factory {
  using Simulator = qsim::Simulator;
  using StateSpace = Simulator::StateSpace;

  // num_sim_threads, num_state_threads and num_dblocks are unused, but kept
  // for consistency with other factories.
  Factory(unsigned num_sim_threads,
          unsigned num_state_threads,
          unsigned num_dblocks) {
    ErrorCheck(cublasCreate(&cublas_handle));
    ErrorCheck(custatevecCreate(&custatevec_handle));
  }

  ~Factory() {
    ErrorCheck(cublasDestroy(cublas_handle));
    ErrorCheck(custatevecDestroy(custatevec_handle));
  }

  StateSpace CreateStateSpace() const {
    return StateSpace(cublas_handle, custatevec_handle);
  }

  Simulator CreateSimulator() const {
    return Simulator(custatevec_handle);
  }

  cublasHandle_t cublas_handle;
  custatevecHandle_t custatevec_handle;
};

inline void SetFlushToZeroAndDenormalsAreZeros() {}
inline void ClearFlushToZeroAndDenormalsAreZeros() {}

}

#include "../pybind_main.cpp"
