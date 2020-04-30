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
#include "../lib/io.h"

namespace qsim {

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
10 fs 0 1 0.2 0.6
11 cp 0 1 0.5
)";

  Circuit<GateQSim<float>> circuit;
  std::stringstream ss1(valid_circuit);

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss1, circuit), true);
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.gates.size(), 18);

  std::stringstream ss2(valid_circuit);

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(4, provider, ss2, circuit), true);
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.gates.size(), 10);
}

// The following tests print error messages if passed. This is okay.

TEST(CircuitQsimParserTest, InvalidGateName) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 badgate 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, TrailingSpace) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cz 0 1 )";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, TrailingCharacters) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1 cc
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, InvalidQubitRange) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 2
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, QubitIsNotNumber) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h i
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, SameQubits) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 0
1 cz 1 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, InvalidSingleQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, InvalidTwoQubitGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h
1 cz 0)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
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

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
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

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
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

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
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

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, InvalidFsimGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 fs 0 1 0.5)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

TEST(CircuitQsimParserTest, InvalidCpGate) {
  constexpr char invalid_circuit[] =
R"(2
0 h 0
0 h 1
1 cp 0 1)";

  std::stringstream ss(invalid_circuit);
  Circuit<GateQSim<float>> circuit;

  EXPECT_EQ(
      CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit), false);
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
