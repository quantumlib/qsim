// Copyright 2026 Google LLC. All Rights Reserved.
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

#include "gtest/gtest.h"

#include "../lib/circuit.h"
#include "../lib/formux.h"
#include "../lib/gates_cirq.h"
#include "../lib/mps_circuit.h"
#include "../lib/operation.h"

namespace qsim {
namespace mps {
namespace {

TEST(MPSCircuit, ApplyXGate) {
  using FP = float;
  using Op = Operation<FP>;
  using SS = MPSStateSpace<For, FP>;

  MPSSimulator<For, FP> simulator(1);
  SS state_space(1);
  auto state = state_space.Create(2, 2);
  state_space.SetStateZero(state);

  Circuit<Op> circuit{2, {}};
  circuit.ops.emplace_back(Cirq::X<FP>::Create(0, 0));

  EXPECT_TRUE(ApplyCircuit(circuit, simulator, state));

  float wave_function[8] = {};
  SS::ToWaveFunction(state, wave_function);
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(wave_function[i], i == 4 ? 1.0 : 0.0, 1e-5)
        << "Unexpected wave-function component at index " << i;
  }
}

TEST(MPSCircuit, RejectMeasurement) {
  using FP = float;
  using Op = Operation<FP>;
  using SS = MPSStateSpace<For, FP>;

  MPSSimulator<For, FP> simulator(1);
  SS state_space(1);
  auto state = state_space.Create(2, 2);
  state_space.SetStateZero(state);

  Circuit<Op> circuit{2, {}};
  circuit.ops.emplace_back(Measurement{});

  EXPECT_THROW(ApplyCircuit(circuit, simulator, state), std::invalid_argument);
}

}  // namespace
}  // namespace mps
}  // namespace qsim
