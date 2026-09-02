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

#include <stdexcept>

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/classical_control_obs.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"
#include "../lib/gate.h"
#include "../lib/operation.h"
#include "../lib/operation_base.h"

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

class CircuitQSimParserTest : public ::testing::Test {
 protected:
  qsim::cc::SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);
  }
};

namespace qsim {

TEST_F(CircuitQSimParserTest, ValidCircuit) {
  constexpr char valid_circuit[] =
R"(2
0 id1 0
0 h 1
1 t 0
1 x 1
2 y 0
2 z 1
3 x_1_2 0
3 y_1_2 1
# comment
4 rx 0 0.7
4 ry 1 0.8
5 rz 0 0.9

5 rxy 1 0.3 0.7
6 hz_1_2 0
7 id2 0 1
8 cz 0 1
9 is 0 1
10 m 0 1
11 fs 0 1 0.2 0.6
12 cp 0 1 0.5
13 m 0
14 m 1
15 p 0.3
)";

  auto circuit1 = TestQSimParser::Run(valid_circuit, 99);
  EXPECT_EQ(circuit1.num_qubits, 2);
  EXPECT_EQ(circuit1.ops.size(), 22);

  auto circuit2 = TestQSimParser::Run(valid_circuit, 4);
  EXPECT_EQ(circuit2.num_qubits, 2);
  EXPECT_EQ(circuit2.ops.size(), 10);
}

TEST_F(CircuitQSimParserTest, ValidCircuitWithControlledGates) {
  constexpr char valid_circuit[] =
R"(5
0 c 0 1 h 2
1 c 4 3 2 is 0 1
2 c 2 4 p 0.5
)";

  using CG = ControlledGate<float>;

  auto circuit = TestQSimParser::Run(valid_circuit, 99);
  EXPECT_EQ(circuit.num_qubits, 5);
  EXPECT_EQ(circuit.ops.size(), 3);
  const auto* pg0 = OpGetAlternative<CG>(circuit.ops[0]);
  const auto* pg1 = OpGetAlternative<CG>(circuit.ops[1]);
  const auto* pg2 = OpGetAlternative<CG>(circuit.ops[2]);
  ASSERT_NE(pg0, nullptr);
  ASSERT_NE(pg1, nullptr);
  ASSERT_NE(pg2, nullptr);
  EXPECT_EQ(pg0->qubits.size(), 1);
  EXPECT_EQ(pg0->controlled_by.size(), 2);
  EXPECT_EQ(pg1->qubits.size(), 2);
  EXPECT_EQ(pg1->controlled_by.size(), 3);
  EXPECT_EQ(pg2->qubits.size(), 0);
  EXPECT_EQ(pg2->controlled_by.size(), 2);
}

TEST_F(CircuitQSimParserTest, ValidTimeOrder) {
  constexpr char valid_circuit[] =
R"(4
0 cz 0 3
1 cz 0 3
2 cz 1 2
3 m 1
3 h 2
4 cz 1 2
5 cz 1 2
6 cz 0 3
7 cz 0 3
8 c 1 x 2
9 h 1
9 h 3
10 h 0
10 h 2
)";

  auto circuit = TestQSimParser::Run(valid_circuit, 99);
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 14);
}

TEST_F(CircuitQSimParserTest, InvalidGateName) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 badgate 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, TrailingCharacters) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1 cc
1 cz 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidQubitRange1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 2
1 cz 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, QubitIsNotNumber1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h i
1 cz 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, SameQubits1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 1 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidSingleQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidTwoQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidRxGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rx 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidRyGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 ry 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidRzGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rz 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidRxyGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rxy 0 0.7)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidFsimGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 fs 0 1 0.5)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidCpGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cp 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, TimeOutOfOrder) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
2 t 0
2 t 1
1 cz 0 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidQubitRange2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 2)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, QubitIsNotNumber2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m i 1)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, SameQubits2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 0)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, NoControlQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c is 0 1
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidControlQubitRange) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 5 is 2 3
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, ControlQubitIsNotNumber) {
  constexpr char invalid_circuit[] =
R"(4
1 c 3 x is 0 1
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, SameControlQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 1 is 2 3
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, SameControlAndTargetQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 2 is 0 1
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, OverlappingQubits1) {
  constexpr char invalid_circuit[] =
R"(4
0 h 0
0 h 1
0 t 0
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, OverlappingQubits2) {
  constexpr char invalid_circuit[] =
R"(4
0 h 0
0 h 1
0 c 0 2 t 3
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidTimeOrder1) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
2 cz 2 3
1 cz 1 2
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidTimeOrder2) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
0 cz 2 3
2 cz 0 3
1 m 1 2
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, InvalidTimeOrder3) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
0 cz 2 3
2 m 0 3
1 cz 1 2
)";

  EXPECT_THROW(TestQSimParser::Run(invalid_circuit, 99), std::invalid_argument);
}

}  // namespace qsim

namespace qsim::cc {
namespace {

TEST_F(CircuitQSimParserTest, ParsesSimpleSymbolDefinitions) {
  std::string_view sym_defs = "nq = 5 c1 = 1 c2 = c1 * 2 f1 = 3.14 * c1";

  ParseSymbols<TestParserError, TestRuntimeError>(sym_defs, symtab);

  const Symbol* nq = symtab.Lookup("nq");
  ASSERT_NE(nq, nullptr);
  EXPECT_EQ(nq->GetInt(), 5);

  const Symbol* c1 = symtab.Lookup("c1");
  ASSERT_NE(c1, nullptr);
  EXPECT_EQ(c1->GetInt(), 1);

  const Symbol* c2 = symtab.Lookup("c2");
  ASSERT_NE(c2, nullptr);
  EXPECT_EQ(c2->GetInt(), 2);

  const Symbol* f1 = symtab.Lookup("f1");
  ASSERT_NE(f1, nullptr);
  EXPECT_TRUE(f1->IsFloat());
  EXPECT_DOUBLE_EQ(f1->GetFloat(), 3.14);
}

TEST_F(CircuitQSimParserTest, ParseSymbolsHandlesSemicolonsAndDelimiters) {
  std::string_view sym_defs = "a = 10; b = a + 5;\n c = b * 2";

  ParseSymbols<TestParserError, TestRuntimeError>(sym_defs, symtab);

  EXPECT_EQ(symtab.Lookup("a")->GetInt(), 10);
  EXPECT_EQ(symtab.Lookup("b")->GetInt(), 15);
  EXPECT_EQ(symtab.Lookup("c")->GetInt(), 30);
}

TEST_F(CircuitQSimParserTest, ParseSymbolsThrowsOnInvalidSyntax) {
  std::string_view bad_defs = "a 10"; // Missing '='

  EXPECT_THROW(
      (ParseSymbols<TestParserError, TestRuntimeError>(bad_defs, symtab)),
      std::invalid_argument);
}

TEST_F(CircuitQSimParserTest, ParsesConstantDeclarationsAndQubitCount) {
  std::string circuit_str = R"(
    2  # 2-qubit circuit
    const float phi = 3.14159 / 2.0
    0 h 0
    1 rx 1 phi
  )";

  auto [circuit, obss] = TestQSimParser::Run(circuit_str,
                                             /*max_depth=*/100, symtab);

  EXPECT_EQ(circuit.num_qubits, 2u);
  EXPECT_TRUE(obss.Empty());

  // Verify symbol table contains constants from circuit
  const Symbol* phi = symtab.Lookup("phi");
  ASSERT_NE(phi, nullptr);
  EXPECT_TRUE(phi->IsFloat());
  EXPECT_TRUE(phi->IsReadOnly());
}

TEST_F(CircuitQSimParserTest, ParsesClassicalControlAndHistograms) {
  std::string circuit_str = R"(
    2
    h 0
    m 0 1 m1
    if m1[0] ^ m1[1]
      x 1
    end
    histogram m1
  )";

  auto [circuit, obss] = TestQSimParser::Run(circuit_str, 100, symtab);

  EXPECT_EQ(circuit.num_qubits, 2u);

  // Verify histogram observable was registered
  const Observable* hist = obss.Lookup("m1");
  ASSERT_NE(hist, nullptr);
  EXPECT_EQ(hist->Size(), 2u); // 2 qubits measured in m1 tag
}

TEST_F(CircuitQSimParserTest, ImplicitTimeTagAssignmentOrdering) {
  // Gates without time tags should automatically increment time step per qubit
  std::string circuit_str = R"(
    2
    h 0
    h 1
    cz 0 1
  )";

  auto [circuit, _] = TestQSimParser::Run(circuit_str, 100, symtab);

  ASSERT_EQ(circuit.ops.size(), 3u);

  auto& bop0 = OpBaseOperation(circuit.ops[0]);
  EXPECT_EQ(bop0.time, 0);
  EXPECT_EQ(bop0.qubits.size(), 1);

  auto& bop1 = OpBaseOperation(circuit.ops[1]);
  EXPECT_EQ(bop1.time, 0);
  EXPECT_EQ(bop1.qubits.size(), 1);

  auto& bop2 = OpBaseOperation(circuit.ops[2]);
  EXPECT_EQ(bop2.time, 1);
  EXPECT_EQ(bop2.qubits.size(), 2);
}

TEST_F(CircuitQSimParserTest, ExpressionQubitIndices) {
  std::string circuit_str = R"(
    5
    const int i = 0
    const int j = 1
    const int k = 2
    const int qs(nq)
    h qs[0] + 1
    h qs[1] - 1
    h qs[2]
    h qs[3]
    h qs[4]
    cz i + 1 j * k
    m i k (i + 3)
    m qs[0] i + k + k m1
    c qs[0] qs[3] + 1 cx qs[1] j + 1
  )";

  auto [circuit, _] = TestQSimParser::Run(circuit_str, 100, symtab);

  ASSERT_EQ(circuit.ops.size(), 9u);

  // h.
  auto& bop0 = OpBaseOperation(circuit.ops[0]);
  EXPECT_EQ(bop0.time, 0);
  ASSERT_EQ(bop0.qubits.size(), 1);
  EXPECT_EQ(bop0.qubits[0], 1);

  // h.
  auto& bop1 = OpBaseOperation(circuit.ops[1]);
  EXPECT_EQ(bop1.time, 0);
  ASSERT_EQ(bop1.qubits.size(), 1);
  EXPECT_EQ(bop1.qubits[0], 0);

  // h.
  auto& bop2 = OpBaseOperation(circuit.ops[2]);
  EXPECT_EQ(bop2.time, 0);
  ASSERT_EQ(bop2.qubits.size(), 1);
  EXPECT_EQ(bop2.qubits[0], 2);

  // h.
  auto& bop3 = OpBaseOperation(circuit.ops[3]);
  EXPECT_EQ(bop3.time, 0);
  ASSERT_EQ(bop3.qubits.size(), 1);
  EXPECT_EQ(bop3.qubits[0], 3);

  // h.
  auto& bop4 = OpBaseOperation(circuit.ops[4]);
  EXPECT_EQ(bop4.time, 0);
  ASSERT_EQ(bop4.qubits.size(), 1);
  EXPECT_EQ(bop4.qubits[0], 4);

  // cz
  auto& bop5 = OpBaseOperation(circuit.ops[5]);
  EXPECT_EQ(bop5.time, 1);
  ASSERT_EQ(bop5.qubits.size(), 2);
  EXPECT_EQ(bop5.qubits[0], 1);
  EXPECT_EQ(bop5.qubits[1], 2);

  // m.
  auto& bop6 = OpBaseOperation(circuit.ops[6]);
  EXPECT_EQ(bop6.time, 2);
  ASSERT_EQ(bop6.qubits.size(), 3);
  EXPECT_EQ(bop6.qubits[0], 0);
  EXPECT_EQ(bop6.qubits[1], 2);
  EXPECT_EQ(bop6.qubits[2], 3);

  // m.
  auto& bop7 = OpBaseOperation(circuit.ops[7]);
  EXPECT_EQ(bop7.time, 3);
  ASSERT_EQ(bop7.qubits.size(), 2);
  EXPECT_EQ(bop7.qubits[0], 0);
  EXPECT_EQ(bop7.qubits[1], 4);

  // c.
  auto* bop8 = OpGetAlternative<ControlledGate<float>>(circuit.ops[8]);
  ASSERT_NE(bop8, nullptr);
  EXPECT_EQ(bop8->time, 4);
  ASSERT_EQ(bop8->qubits.size(), 2);
  EXPECT_EQ(bop8->qubits[0], 1);
  EXPECT_EQ(bop8->qubits[1], 2);
  ASSERT_EQ(bop8->controlled_by.size(), 2);
  EXPECT_EQ(bop8->controlled_by[0], 0);
  EXPECT_EQ(bop8->controlled_by[1], 4);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
