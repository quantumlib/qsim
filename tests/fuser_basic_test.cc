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

#include <vector>
#include <sstream>

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"

namespace qsim {

struct IO {
  static void errorf(const char* format, ...) {}
  static void messagef(const char* format, ...) {}
};

constexpr char provider[] = "fuser_basic_test";

constexpr char circuit_string1[] =
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
5 cz 2 1
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

TEST(FuserBasicTest, NoTimesToSplitAt) {
  std::stringstream ss(circuit_string1);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 27);

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 5);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 6);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 3);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[5]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 6);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[1].gates[5]->time, 3);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 9);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 5);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[5]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[5]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[6]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[6]->time, 7);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[8]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[8]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits[0], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 9);
  EXPECT_EQ(fused_gates[3].qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates.size(), 3);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 1);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 3);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits[0], 3);
}


TEST(FuserBasicTest, TimesToSplitAt1) {
  std::stringstream ss(circuit_string1);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 27);

  std::vector<unsigned> times_to_split_at{3, 8, 10};

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 6);


  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 6);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 3);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[5]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 6);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[1].gates[5]->time, 3);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 5);
  EXPECT_EQ(fused_gates[3].qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 1);
  EXPECT_EQ(fused_gates[3].qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates.size(), 8);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 5);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates[3]->kind, kGateX);
  EXPECT_EQ(fused_gates[3].gates[3]->time, 6);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[3].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[3].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[4]->qubits[0], 2);
  EXPECT_EQ(fused_gates[3].gates[5]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[5]->time, 7);
  EXPECT_EQ(fused_gates[3].gates[5]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[5]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[5]->qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates[6]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[6]->time, 8);
  EXPECT_EQ(fused_gates[3].gates[6]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[3].gates[7]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[7]->qubits[0], 2);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 0);
  EXPECT_EQ(fused_gates[4].qubits[1], 1);
  EXPECT_EQ(fused_gates[4].gates.size(), 3);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 1);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits[0], 1);

  EXPECT_EQ(fused_gates[5].kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].time, 9);
  EXPECT_EQ(fused_gates[5].qubits.size(), 2);
  EXPECT_EQ(fused_gates[5].qubits[0], 2);
  EXPECT_EQ(fused_gates[5].qubits[1], 3);
  EXPECT_EQ(fused_gates[5].gates.size(), 3);
  EXPECT_EQ(fused_gates[5].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[1], 3);
  EXPECT_EQ(fused_gates[5].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[5].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[5].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[5].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits[0], 3);
}

TEST(FuserBasicTest, TimesToSplitAt2) {
  std::stringstream ss(circuit_string1);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 27);

  std::vector<unsigned> times_to_split_at{2, 10};

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 5);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 5);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 5);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 9);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 5);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[5]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[5]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[6]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[6]->time, 7);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[8]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[8]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits[0], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 9);
  EXPECT_EQ(fused_gates[3].qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates.size(), 4);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateY);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 9);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[3]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[3]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits[0], 1);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 4);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 3);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[3]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[3]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[3]->qubits[0], 3);
}

constexpr char circuit_string2[] =
R"(3
0 h 0
0 h 1
0 h 2
1 cz 0 1
2 t 0
2 x 1
2 y 2
3 cz 1 2
4 x 0
)";

TEST(FuserBasicTest, OrphanedQubits1) {
  std::stringstream ss(circuit_string2);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(2, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.gates.size(), 7);

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 2);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 5);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 2);
}

TEST(FuserBasicTest, OrphanedQubits2) {
  std::stringstream ss(circuit_string2);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.gates.size(), 9);

  std::vector<unsigned> times_to_split_at{1, 4};

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 4);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 3);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 3);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[1], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateT);
  EXPECT_EQ(fused_gates[3].time, 2);
  EXPECT_EQ(fused_gates[3].qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateX);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 0);
}

TEST(FuserBasicTest, UnfusibleSingleQubitGate) {
  std::stringstream ss(circuit_string2);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(2, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.gates.size(), 7);

  circuit.gates[1].unfusible = true;
  circuit.gates[2].unfusible = true;

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 3);

  EXPECT_EQ(fused_gates[0].kind, kGateHd);
  EXPECT_EQ(fused_gates[0].time, 0);
  EXPECT_EQ(fused_gates[0].qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 1);
  EXPECT_EQ(fused_gates[2].qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 0);
  EXPECT_EQ(fused_gates[2].qubits[1], 1);
  EXPECT_EQ(fused_gates[2].gates.size(), 4);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[1], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[0], 1);
}

constexpr char circuit_string3[] =
R"(4
0 h 0
0 h 1
0 h 2
0 h 3
1 cz 0 1
1 m 2
1 m 3
2 x 0
2 y 1
2 is 2 3
3 cz 0 1
3 m 2 3
4 x 0
4 y 1
4 is 2 3
5 m 2 3
5 m 0 1
)";

TEST(FuserBasicTest, MeasurementGate) {
  std::stringstream ss(circuit_string3);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 17);

  // Vector of pointers to gates.
  std::vector<const GateQSim<float>*> pgates;
  pgates.reserve(circuit.gates.size());

  for (const auto& gate : circuit.gates) {
    pgates.push_back(&gate);
  }

  using Fuser = BasicGateFuser<IO, const GateQSim<float>*>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, pgates);

  EXPECT_EQ(fused_gates.size(), 11);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 3);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateHd);
  EXPECT_EQ(fused_gates[2].time, 0);
  EXPECT_EQ(fused_gates[2].qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].qubits[0], 3);
  EXPECT_EQ(fused_gates[2].gates.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 3);

  EXPECT_EQ(fused_gates[3].kind, kMeasurement);
  EXPECT_EQ(fused_gates[3].time, 1);
  EXPECT_EQ(fused_gates[3].qubits.size(), 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 2);
  EXPECT_EQ(fused_gates[3].qubits[1], 3);
  EXPECT_EQ(fused_gates[3].gates.size(), 2);

  EXPECT_EQ(fused_gates[4].kind, kGateIS);
  EXPECT_EQ(fused_gates[4].time, 2);
  EXPECT_EQ(fused_gates[4].qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateIS);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 3);

  EXPECT_EQ(fused_gates[5].kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].time, 3);
  EXPECT_EQ(fused_gates[5].qubits.size(), 2);
  EXPECT_EQ(fused_gates[5].qubits[0], 0);
  EXPECT_EQ(fused_gates[5].qubits[1], 1);
  EXPECT_EQ(fused_gates[5].gates.size(), 3);
  EXPECT_EQ(fused_gates[5].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[5].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[5].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[5].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[5].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].gates[2]->time, 3);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[6].kind, kMeasurement);
  EXPECT_EQ(fused_gates[6].time, 3);
  EXPECT_EQ(fused_gates[6].qubits.size(), 2);
  EXPECT_EQ(fused_gates[6].qubits[0], 2);
  EXPECT_EQ(fused_gates[6].qubits[1], 3);
  EXPECT_EQ(fused_gates[6].gates.size(), 1);

  EXPECT_EQ(fused_gates[7].kind, kGateIS);
  EXPECT_EQ(fused_gates[7].time, 4);
  EXPECT_EQ(fused_gates[7].qubits.size(), 2);
  EXPECT_EQ(fused_gates[7].qubits[0], 2);
  EXPECT_EQ(fused_gates[7].qubits[1], 3);
  EXPECT_EQ(fused_gates[7].gates.size(), 1);
  EXPECT_EQ(fused_gates[7].gates[0]->kind, kGateIS);
  EXPECT_EQ(fused_gates[7].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits[1], 3);

  EXPECT_EQ(fused_gates[8].kind, kGateX);
  EXPECT_EQ(fused_gates[8].time, 4);
  EXPECT_EQ(fused_gates[8].qubits.size(), 1);
  EXPECT_EQ(fused_gates[8].qubits[0], 0);
  EXPECT_EQ(fused_gates[8].gates.size(), 1);
  EXPECT_EQ(fused_gates[8].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[8].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[8].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[8].gates[0]->qubits[0], 0);

  EXPECT_EQ(fused_gates[9].kind, kGateY);
  EXPECT_EQ(fused_gates[9].time, 4);
  EXPECT_EQ(fused_gates[9].qubits.size(), 1);
  EXPECT_EQ(fused_gates[9].qubits[0], 1);
  EXPECT_EQ(fused_gates[9].gates.size(), 1);
  EXPECT_EQ(fused_gates[9].gates[0]->kind, kGateY);
  EXPECT_EQ(fused_gates[9].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[9].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[9].gates[0]->qubits[0], 1);

  EXPECT_EQ(fused_gates[10].kind, kMeasurement);
  EXPECT_EQ(fused_gates[10].time, 5);
  EXPECT_EQ(fused_gates[10].qubits.size(), 4);
  EXPECT_EQ(fused_gates[10].qubits[0], 2);
  EXPECT_EQ(fused_gates[10].qubits[1], 3);
  EXPECT_EQ(fused_gates[10].qubits[2], 0);
  EXPECT_EQ(fused_gates[10].qubits[3], 1);
  EXPECT_EQ(fused_gates[10].gates.size(), 2);
}

constexpr char circuit_string4[] =
R"(5
0 h 0
0 h 1
0 h 2
0 h 3
0 h 4
1 cz 0 1
1 cz 2 3
2 c 0 1 4 h 2
3 h 0
3 h 1
3 h 2
3 h 3
3 h 4
)";

TEST(FuserBasicTest, ControlledGate) {
  std::stringstream ss(circuit_string4);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 5);
  EXPECT_EQ(circuit.gates.size(), 13);

  // Vector of pointers to gates.
  std::vector<const GateQSim<float>*> pgates;
  pgates.reserve(circuit.gates.size());

  for (const auto& gate : circuit.gates) {
    pgates.push_back(&gate);
  }

  using Fuser = BasicGateFuser<IO, const GateQSim<float>*>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, pgates);

  EXPECT_EQ(fused_gates.size(), 8);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 3);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 4);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 3);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateHd);
  EXPECT_EQ(fused_gates[2].time, 0);
  EXPECT_EQ(fused_gates[2].qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].qubits[0], 4);
  EXPECT_EQ(fused_gates[2].gates.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 4);

  EXPECT_EQ(fused_gates[3].kind, kGateHd);
  EXPECT_EQ(fused_gates[3].time, 2);
  EXPECT_EQ(fused_gates[3].qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].qubits[0], 2);
  EXPECT_EQ(fused_gates[3].parent->controlled_by.size(), 3);
  EXPECT_EQ(fused_gates[3].parent->controlled_by[0], 0);
  EXPECT_EQ(fused_gates[3].parent->controlled_by[1], 1);
  EXPECT_EQ(fused_gates[3].parent->controlled_by[2], 4);
  EXPECT_EQ(fused_gates[3].parent->cmask, 7);
  EXPECT_EQ(fused_gates[3].gates.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[4].kind, kGateHd);
  EXPECT_EQ(fused_gates[4].time, 3);
  EXPECT_EQ(fused_gates[4].qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].qubits[0], 0);
  EXPECT_EQ(fused_gates[4].gates.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 0);

  EXPECT_EQ(fused_gates[5].kind, kGateHd);
  EXPECT_EQ(fused_gates[5].time, 3);
  EXPECT_EQ(fused_gates[5].qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].qubits[0], 1);
  EXPECT_EQ(fused_gates[5].gates.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[5].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[0], 1);

  EXPECT_EQ(fused_gates[6].kind, kGateHd);
  EXPECT_EQ(fused_gates[6].time, 3);
  EXPECT_EQ(fused_gates[6].qubits.size(), 1);
  EXPECT_EQ(fused_gates[6].qubits[0], 2);
  EXPECT_EQ(fused_gates[6].gates.size(), 1);
  EXPECT_EQ(fused_gates[6].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[6].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[6].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[6].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[7].kind, kGateHd);
  EXPECT_EQ(fused_gates[7].time, 3);
  EXPECT_EQ(fused_gates[7].qubits.size(), 1);
  EXPECT_EQ(fused_gates[7].qubits[0], 4);
  EXPECT_EQ(fused_gates[7].gates.size(), 1);
  EXPECT_EQ(fused_gates[7].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[7].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits.size(), 1);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits[0], 4);
}

namespace {

template <typename Gate, typename FusedGate>
bool TestFusedGates(unsigned num_qubits,
                    const std::vector<Gate>& gates,
                    const std::vector<FusedGate>& fused_gates) {
  std::vector<unsigned> times(num_qubits, 0);
  std::vector<unsigned> gate_map(gates.size(), 0);

  // Test if gate times are ordered correctly.
  for (auto g : fused_gates) {
    if (g.parent->controlled_by.size() > 0 && g.gates.size() > 1) {
      return false;
    }

    for (auto p : g.gates) {
      auto k = (std::size_t(p) - std::size_t(gates.data())) / sizeof(*p);

      if (k >= gate_map.size()) {
        return false;
      }

      ++gate_map[k];

      if (p->kind == gate::kMeasurement) {
        if (g.parent->kind != gate::kMeasurement || g.parent->time != p->time) {
          return false;
        }
      }

      for (auto q : p->qubits) {
        if (p->time < times[q]) {
          return false;
        }
        times[q] = p->time;
      }

      for (auto q : p->controlled_by) {
        if (p->time < times[q]) {
          return false;
        }
        times[q] = p->time;
      }
    }
  }

  // Test if all gates are present only once.
  for (auto m : gate_map) {
    if (m != 1) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(FuserBasicTest, ValidTimeOrder) {
  using Gate = GateQSim<float>;
  using Fuser = BasicGateFuser<IO, Gate>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 8;
    auto gate1 = GateZ<float>::Create(1, 2);
    auto gate2 = GateZ<float>::Create(2, 5);

    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      MakeControlledGate({1}, gate1),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(2, 0, 1),
      GateCZ<float>::Create(1, 3, 4),
      GateCZ<float>::Create(2, 2, 3),
      GateCZ<float>::Create(3, 1, 2),
      MakeControlledGate({4}, gate2),
      GateCZ<float>::Create(3, 3, 4),
      GateCZ<float>::Create(5, 0, 1),
      GateCZ<float>::Create(4, 2, 3),
      GateCZ<float>::Create(5, 4, 5),
      GateCZ<float>::Create(4, 6, 7),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 14);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(1, 1, 2),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(1, 3, 4),
      gate::Measurement<Gate>::Create(2, {0, 1, 2}),
      gate::Measurement<Gate>::Create(2, {4, 5}),
      GateCZ<float>::Create(3, 0, 1),
      GateCZ<float>::Create(3, 2, 3),
      GateCZ<float>::Create(4, 1, 2),
      GateCZ<float>::Create(3, 4, 5),
      GateCZ<float>::Create(4, 3, 4),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 11);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(1, 1, 2),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(1, 3, 4),
      GateCZ<float>::Create(1, 5, 0),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 6);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 8;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(1, 1, 2),
      GateCZ<float>::Create(1, 3, 4),
      GateCZ<float>::Create(0, 6, 7),
      GateCZ<float>::Create(1, 5, 6),
      GateCZ<float>::Create(1, 7, 0),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 8);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 1, 2),
      GateCZ<float>::Create(1, 0, 3),
      GateCZ<float>::Create(3, 1, 2),
      GateCZ<float>::Create(3, 0, 3),
      GateCZ<float>::Create(5, 1, 2),
      GateCZ<float>::Create(4, 0, 3),
    };

    std::vector<unsigned> time_boundary = {3};
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), time_boundary);

    EXPECT_EQ(fused_gates.size(), 6);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 1, 2),
      GateCZ<float>::Create(1, 0, 3),
      gate::Measurement<Gate>::Create(3, {1, 2}),
      GateCZ<float>::Create(3, 0, 3),
      GateCZ<float>::Create(5, 1, 2),
      GateCZ<float>::Create(4, 0, 3),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 7);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }
}

TEST(FuserBasicTest, InvalidTimeOrder) {
  using Gate = GateQSim<float>;
  using Fuser = BasicGateFuser<IO, Gate>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 1, 2),
      GateCZ<float>::Create(1, 0, 2),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      GateCZ<float>::Create(1, 1, 2),
    };

    std::vector<unsigned> time_boundary = {1};
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), time_boundary);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      GateCZ<float>::Create(1, 1, 2),
    };

    std::vector<unsigned> time_boundary = {2};
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), time_boundary);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      gate::Measurement<Gate>::Create(2, {0, 3}),
      GateCZ<float>::Create(1, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      gate::Measurement<Gate>::Create(1, {1, 2}),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    auto gate = GateZ<float>::Create(1, 1);

    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      MakeControlledGate({3}, gate),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    auto gate = GateZ<float>::Create(2, 1);

    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      MakeControlledGate({3}, gate),
      GateCZ<float>::Create(1, 0, 3),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

TEST(FuserBasicTest, QubitsOutOfRange) {
  using Gate = GateQSim<float>;
  using Fuser = BasicGateFuser<IO, Gate>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 3),
      GateCZ<float>::Create(0, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 3;
    auto gate = GateZ<float>::Create(0, 2);
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      MakeControlledGate({3}, gate),
    };

    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
