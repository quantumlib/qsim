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

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/expect.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/simmux.h"

namespace qsim {

TEST(ExpectTest, ExpectationValue) {
  using StateSpace = Simulator<For>::StateSpace;
  using State = StateSpace::State;
  using fp_type = StateSpace::fp_type;
  using Fuser = MultiQubitGateFuser<IO, GateQSim<fp_type>>;

  unsigned num_qubits = 16;
  unsigned depth = 16;

  StateSpace state_space(1);
  Simulator<For> simulator(1);

  State state = state_space.Create(num_qubits);
  state_space.SetStateZero(state);

  std::vector<GateQSim<fp_type>> circuit;
  circuit.reserve(num_qubits * (3 * depth / 2 + 1));

  for (unsigned k = 0; k < num_qubits; ++k) {
    circuit.push_back(GateHd<fp_type>::Create(0, k));
  }

  unsigned t = 1;

  for (unsigned i = 0; i < depth / 2; ++i) {
    for (unsigned k = 0; k < num_qubits; ++k) {
      circuit.push_back(GateRX<fp_type>::Create(t, k, 0.1 * k));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits / 2; ++k) {
      circuit.push_back(GateIS<fp_type>::Create(t, 2 * k, 2 * k + 1));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits; ++k) {
      circuit.push_back(GateRY<fp_type>::Create(t, k, 0.1 * k));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits / 2; ++k) {
      circuit.push_back(GateIS<fp_type>::Create(
          t, 2 * k + 1, (2 * k + 2) % num_qubits));
    }

    ++t;
  }

  for (const auto& gate : circuit) {
    ApplyGate(simulator, gate, state);
  }

  fp_type expected_real[6] = {
    0.014314421865856278,
    0.021889885055134076,
    -0.006954622792545706,
    0.013091871136566622,
    0.004322795104235413,
    -0.008040613483171907,
  };

  Fuser::Parameter param;
  param.max_fused_size = 4;

  State tmp_state = state_space.Null();

  for (unsigned k = 1; k <= 6; ++k) {
    std::vector<OpString<GateQSim<fp_type>>> strings;
    strings.reserve(num_qubits);

    for (unsigned i = 0; i <= num_qubits - k; ++i) {
      strings.push_back({{0.1 + 0.2 * i, 0}, {}});

      strings.back().ops.reserve(k);

      for (unsigned j = 0; j < k; ++j) {
        switch (j % 3) {
        case 0:
          strings.back().ops.push_back(GateX<fp_type>::Create(0, i + j));
          break;
        case 1:
          strings.back().ops.push_back(GateY<fp_type>::Create(0, i + j));
          break;
        case 2:
          strings.back().ops.push_back(GateZ<fp_type>::Create(0, i + j));
          break;
        }
      }
    }

    if (k == 2) {
      tmp_state = state_space.Create(num_qubits - 2);
    }

    auto evala = ExpectationValue<IO, Fuser>(param, strings, state_space,
                                             simulator, state, tmp_state);

    EXPECT_NEAR(std::real(evala), expected_real[k - 1], 1e-6);
    EXPECT_NEAR(std::imag(evala), 0, 1e-8);

    auto evalb = ExpectationValue<IO, Fuser>(strings, simulator, state);

    EXPECT_NEAR(std::real(evalb), expected_real[k - 1], 1e-6);
    EXPECT_NEAR(std::imag(evalb), 0, 1e-8);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
