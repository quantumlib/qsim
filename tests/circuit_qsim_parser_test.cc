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

#include <sstream>

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/gates_qsim.h"

namespace qsim {

struct IO {
  static void errorf(const char* format, ...) {}
  static void messagef(const char* format, ...) {}
};

constexpr char provider[] = "circuit_qsim_parser_test";

TEST(CircuitQsimParserTest, ValidCircuit) {
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
)";

  Circuit<GateQSim<float>> circuit;
  std::stringstream ss1(valid_circuit);

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss1, circuit));
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.gates.size(), 21);

  std::stringstream ss2(valid_circuit);

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(4, provider, ss2, circuit));
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.gates.size(), 10);
}

TEST(CircuitQsimParserTest, ValidCircuitWithControlledGates) {
  constexpr char valid_circuit[] =
R"(5
0 c 0 1 h 2
1 c 4 3 2 is 0 1
)";

  Circuit<GateQSim<float>> circuit;
  std::stringstream ss(valid_circuit);
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 5);
  EXPECT_EQ(circuit.gates.size(), 2);
  EXPECT_EQ(circuit.gates[0].qubits.size(), 1);
  EXPECT_EQ(circuit.gates[0].controlled_by.size(), 2);
  EXPECT_EQ(circuit.gates[1].qubits.size(), 2);
  EXPECT_EQ(circuit.gates[1].controlled_by.size(), 3);
}

TEST(CircuitQsimParserTest, ValidTimeOrder) {
  constexpr char valid_circuit[] =
R"(4
0 cz 0 3
2 cz 1 2
1 cz 0 3
3 m 1
3 h 2
4 cz 1 2
6 cz 0 3
5 cz 1 2
8 c 1 x 2
7 cz 0 3
10 h 0
9 h 1
10 h 2
9 h 3
)";

  Circuit<GateQSim<float>> circuit;
  std::stringstream ss(valid_circuit);
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 14);
}

TEST(CircuitQsimParserTest, InvalidGateName) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 badgate 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, TrailingSpace1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1 )";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, TrailingCharacters) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1 cc
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidQubitRange1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 2
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, QubitIsNotNumber1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h i
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, SameQubits1) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 0
1 cz 1 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidSingleQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidTwoQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidRxGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rx 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidRyGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 ry 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidRzGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rz 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidRxyGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 rxy 0 0.7)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidFsimGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 fs 0 1 0.5)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidCpGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cp 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, TimeOutOfOrder) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 t 0
2 t 1
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidQubitRange2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 2)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, QubitIsNotNumber2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 i)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, TrailingSpace2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 )";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, SameQubits2) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 m 0 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, NoControlQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c is 0 1
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidControlQubitRange) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 5 is 2 3
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, ControlQubitIsNotNumber) {
  constexpr char invalid_circuit[] =
R"(4
1 c 3 x is 0 1
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, SameControlQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 1 is 2 3
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, SameControlAndTargetQubits) {
  constexpr char invalid_circuit[] =
R"(4
0 c 1 2 is 0 1
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, OverlappingQubits1) {
  constexpr char invalid_circuit[] =
R"(4
0 h 0
0 h 1
0 t 0
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, OverlappingQubits2) {
  constexpr char invalid_circuit[] =
R"(4
0 h 0
0 h 1
0 c 0 2 t 3
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidTimeOrder1) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
2 cz 2 3
1 cz 1 2
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidTimeOrder2) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
0 cz 2 3
2 cz 0 3
1 m 1 2
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

TEST(CircuitQsimParserTest, InvalidTimeOrder3) {
  constexpr char invalid_circuit[] =
R"(4
0 cz 0 1
0 cz 2 3
2 m 0 3
1 cz 1 2
)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_FALSE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
