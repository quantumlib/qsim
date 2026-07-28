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
#include "../lib/classical_control_obs.h"
#include "../lib/classical_control_symtab.h"
#include "../lib/formux.h"
#include "../lib/fuser_basic.h"
#include "../lib/io.h"
#include "../lib/operation.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"

struct TestParserError {
  template <typename... Args>
  [[noreturn]] static void Throw(
      std::string_view msg, unsigned lc, Args&&... args) {
    throw std::invalid_argument("syntax error");
  }
};

struct TestRuntimeError {
  template <typename... Args>
  [[noreturn]] static void Throw(
      std::string_view msg, unsigned lc, Args&&... args) {
    throw std::runtime_error("runtime error");
  }
};

using TestQSimParser =
    qsim::CircuitQSimParser<float, TestParserError, TestRuntimeError>;

namespace qsim {

constexpr char circuit_string[] =
R"(4
h 0
h 1
h 2
h 3
cz 0 1
cz 2 3
t 0
x 1
y 2
t 3
y 0
cz 1 2
x 3
t 1
t 2
cz 1 2
x 1
y 2
cz 1 2
t 1
t 2
cz 0 1
cz 2 3
h 0
h 1
h 2
h 3
)";

struct Factory {
  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;

  static StateSpace CreateStateSpace() {
    return StateSpace(1);
  }

  static Simulator CreateSimulator() {
    return Simulator(1);
  }
};

TEST(RunQSimTest, QSimRunner) {
  auto circuit = CircuitQSimParser<float>::Run(circuit_string, 99);

  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 27);

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<BasicGateFuser<IO>>;

  Simulator simulator = Factory::CreateSimulator();
  StateSpace state_space = Factory::CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.verbosity = 0;

  cc::SymTable symtab;
  cc::Observables obss;

  EXPECT_TRUE(Runner::Run(
      param, circuit, state_space, simulator, 1, state, symtab, obss));

  // Calculate entropy.

  float entropy = 0;
  auto size = uint64_t{1} << circuit.num_qubits;

  for (uint64_t i = 0; i < size; ++i) {
    auto ampl = state_space.GetAmpl(state, i);
    float p = std::norm(ampl);
    entropy -= p * std::log(p);
  }

  EXPECT_NEAR(entropy, 2.2192848, 1e-6);
}

TEST(RunQSimTest, CirqGates) {
  auto circuit = CirqCircuit1::GetCircuit<float>(true);
  const auto& expected_results = CirqCircuit1::expected_results1;

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<BasicGateFuser<IO>>;

  Simulator simulator = Factory::CreateSimulator();
  StateSpace state_space = Factory::CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  auto size = uint64_t{1} << circuit.num_qubits;

  EXPECT_FALSE(state_space.IsNull(state));
  EXPECT_EQ(size, expected_results.size());

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.verbosity = 0;

  cc::SymTable symtab;
  cc::Observables obss;

  EXPECT_TRUE(Runner::Run(
      param, circuit, state_space, simulator, 1, state, symtab, obss));

  for (uint64_t i = 0; i < size; ++i) {
    auto ampl = state_space.GetAmpl(state, i);
    EXPECT_NEAR(std::real(ampl), std::real(expected_results[i]), 2e-6);
    EXPECT_NEAR(std::imag(ampl), std::imag(expected_results[i]), 2e-6);
  }
}

TEST(RunQSimTest, Factorial) {
  std::string circuit_str = R"(
    int f = 1
    int i = 1
    repeat i <= n
      f = f * i
      i = i + 1
    end
  )";

  cc::SymTable symtab;
  symtab.EnterScope(symtab.AddScope());
  symtab.Insert("n", cc::Symbol::Int(12));

  cc::Observables obss;

  auto [circuit, _] = TestQSimParser::Run(circuit_str, 100, symtab);

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<BasicGateFuser<IO>>;

  Simulator simulator = Factory::CreateSimulator();
  StateSpace state_space = Factory::CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  Runner::Parameter param;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(
      param, circuit, state_space, simulator, 1, state, symtab, obss));

  const auto* var = symtab.Lookup("f");

  ASSERT_NE(var, nullptr);
  EXPECT_EQ(var->GetInt(), 479001600);
}

TEST(RunQSimTest, Histogram) {
  std::string circuit_str = R"(
    2
    x 1
    m 0 1 m1
    histogram m1
  )";

  cc::SymTable symtab;

  auto [circuit, obss] = TestQSimParser::Run(circuit_str, 100, symtab);

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<BasicGateFuser<IO>>;

  Simulator simulator = Factory::CreateSimulator();
  StateSpace state_space = Factory::CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  Runner::Parameter param;
  param.verbosity = 0;

  for (uint64_t r = 0; r < 100; ++r) {
    state_space.SetStateZero(state);
    EXPECT_TRUE(Runner::Run(
        param, circuit, state_space, simulator, 1, state, symtab, obss));
    obss.Iterate([](auto, auto& obs) { obs.Update(); });
  }

  const auto* h = obss.Lookup("m1");
  ASSERT_NE(h, nullptr);

  EXPECT_EQ(h->num_qubits, 2);

  ASSERT_EQ(h->cur_count.size(), 4);
  EXPECT_EQ(h->cur_count[0], 0);
  EXPECT_EQ(h->cur_count[1], 0);
  EXPECT_EQ(h->cur_count[2], 0);
  EXPECT_EQ(h->cur_count[3], 0);

  ASSERT_EQ(h->total_count.size(), 4);
  EXPECT_EQ(h->total_count[0], 0);
  EXPECT_EQ(h->total_count[1], 0);
  EXPECT_EQ(h->total_count[2], 100);
  EXPECT_EQ(h->total_count[3], 0);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
