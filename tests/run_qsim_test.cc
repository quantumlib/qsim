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
#include <complex>
#include <cstdint>
#include <sstream>

#include "gates_cirq_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"

namespace qsim {

constexpr char provider[] = "run_qsim_test";

constexpr char circuit_string[] =
R"(4
0 h 0
0 h 1
0 h 2
0 h 3
1 cz 0 1
1 cz 2 3
2 t 0
2 x 1
2 y 2
2 t 3
3 y 0
3 cz 1 2
3 x 3
4 t 1
4 t 2
5 cz 1 2
6 x 1
6 y 2
7 cz 1 2
8 t 1
8 t 2
9 cz 0 1
9 cz 2 3
10 h 0
10 h 1
10 h 2
10 h 3
)";

TEST(RunQSimTest, QSimRunner1) {
  std::stringstream ss(circuit_string);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 27);

  using Simulator = Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<float>>, Simulator>;

  float entropy = 0;

  auto measure = [&entropy](
      unsigned k, const StateSpace& state_space, const State& state) {
    // Calculate entropy.

    entropy = 0;

    for (uint64_t i = 0; i < state_space.Size(); ++i) {
      auto ampl = state_space.GetAmpl(state, i);
      float p = std::norm(ampl);
      entropy -= p * std::log(p);
    }
  };

  Runner::Parameter param;
  param.seed = 1;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, circuit, measure));

  EXPECT_NEAR(entropy, 2.2192848, 1e-6);
}

TEST(RunQSimTest, QSimRunner2) {
  std::stringstream ss(circuit_string);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 27);

  using Simulator = Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<float>>, Simulator>;

  StateSpace state_space(circuit.num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.seed = 1;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, circuit, state));

  // Calculate entropy.

  float entropy = 0;

  for (uint64_t i = 0; i < state_space.Size(); ++i) {
    auto ampl = state_space.GetAmpl(state, i);
    float p = std::norm(ampl);
    entropy -= p * std::log(p);
  }

  EXPECT_NEAR(entropy, 2.2192848, 1e-6);
}

constexpr char sample_circuit_string[] = 
R"(2
0 h 0
0 x 1
1 m 1
2 cx 0 1
3 m 0 1
4 m 0
5 cx 1 0
6 m 0
7 x 0
7 h 1
8 m 0 1
)";

TEST(RunQSimTest, QSimSampler) {
  std::stringstream ss(sample_circuit_string);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.gates.size(), 11);

  using Simulator = Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using Result = StateSpace::MeasurementResult;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<float>>, Simulator>;

  StateSpace state_space(circuit.num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  std::vector<Result> results;

  Runner::Parameter param;
  param.seed = 1;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, circuit, state, results));

  // Results should contain (qubit @ time):
  // (1 @ 1) - should be |01)
  EXPECT_TRUE(results[0].bitstring[0]);
  // (0 @ 3), (1 @ 3) - either |01) or |10)
  EXPECT_EQ(results[1].bitstring[0], !results[1].bitstring[1]);
  // (0 @ 4) - should match (0 @ 3)
  EXPECT_EQ(results[1].bitstring[0], results[2].bitstring[0]);
  // (0 @ 6) - either |11) or |10)
  EXPECT_TRUE(results[3].bitstring[0]);
  // (0 @ 8), (1 @ 8) - should be |00)
  EXPECT_FALSE(results[4].bitstring[0]);
  EXPECT_FALSE(results[4].bitstring[1]);
}

TEST(RunQSimTest, CirqGates) {
  auto circuit = CirqCircuit1::GetCircuit<float>();
  const auto& expected_results = CirqCircuit1::expected_results;

  using Simulator = Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, Cirq::GateCirq<float>>,
                            Simulator>;

  StateSpace state_space(circuit.num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_FALSE(state_space.IsNull(state));
  EXPECT_EQ(state_space.Size(), expected_results.size());

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.seed = 1;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, circuit, state));

  for (uint64_t i = 0; i < state_space.Size(); ++i) {
    auto ampl = state_space.GetAmpl(state, i);
    EXPECT_NEAR(std::real(ampl), std::real(expected_results[i]), 2e-6);
    EXPECT_NEAR(std::imag(ampl), std::imag(expected_results[i]), 2e-6);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
