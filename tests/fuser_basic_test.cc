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
#include "../lib/io.h"

namespace qsim {

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
  auto fused_gates = Fuser::FuseGates(circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 5);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 6);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 3);
  EXPECT_EQ(fused_gates[0].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[5]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].num_qubits, 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 6);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[1].gates[5]->time, 3);
  EXPECT_EQ(fused_gates[1].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].num_qubits, 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 9);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 5);
  EXPECT_EQ(fused_gates[2].gates[3]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[5]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[5]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[6]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[6]->time, 7);
  EXPECT_EQ(fused_gates[2].gates[6]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[7]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[8]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[8]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[8]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits[0], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 9);
  EXPECT_EQ(fused_gates[3].num_qubits, 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates.size(), 3);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[3].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 1);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].num_qubits, 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 3);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->num_qubits, 1);
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
  auto fused_gates = Fuser::FuseGates(
      circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 6);


  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 6);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 3);
  EXPECT_EQ(fused_gates[0].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[5]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[5]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].num_qubits, 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 6);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[5]->kind, kGateX);
  EXPECT_EQ(fused_gates[1].gates[5]->time, 3);
  EXPECT_EQ(fused_gates[1].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[5]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].num_qubits, 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 5);
  EXPECT_EQ(fused_gates[3].num_qubits, 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 1);
  EXPECT_EQ(fused_gates[3].qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates.size(), 8);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 5);
  EXPECT_EQ(fused_gates[3].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates[3]->kind, kGateX);
  EXPECT_EQ(fused_gates[3].gates[3]->time, 6);
  EXPECT_EQ(fused_gates[3].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[4]->kind, kGateY);
  EXPECT_EQ(fused_gates[3].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[3].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[4]->qubits[0], 2);
  EXPECT_EQ(fused_gates[3].gates[5]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[5]->time, 7);
  EXPECT_EQ(fused_gates[3].gates[5]->num_qubits, 2);
  EXPECT_EQ(fused_gates[3].gates[5]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[5]->qubits[1], 2);
  EXPECT_EQ(fused_gates[3].gates[6]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[6]->time, 8);
  EXPECT_EQ(fused_gates[3].gates[6]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[3].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[3].gates[7]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[7]->qubits[0], 2);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].num_qubits, 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 0);
  EXPECT_EQ(fused_gates[4].qubits[1], 1);
  EXPECT_EQ(fused_gates[4].gates.size(), 3);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 1);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits[0], 1);

  EXPECT_EQ(fused_gates[5].kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].time, 9);
  EXPECT_EQ(fused_gates[5].num_qubits, 2);
  EXPECT_EQ(fused_gates[5].qubits[0], 2);
  EXPECT_EQ(fused_gates[5].qubits[1], 3);
  EXPECT_EQ(fused_gates[5].gates.size(), 3);
  EXPECT_EQ(fused_gates[5].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].gates[0]->time, 9);
  EXPECT_EQ(fused_gates[5].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[1], 3);
  EXPECT_EQ(fused_gates[5].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[5].gates[1]->time, 10);
  EXPECT_EQ(fused_gates[5].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[5].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[5].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[5].gates[2]->num_qubits, 1);
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
  auto fused_gates = Fuser::FuseGates(
      circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 5);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 5);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].time, 1);
  EXPECT_EQ(fused_gates[1].num_qubits, 2);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates.size(), 5);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 3);
  EXPECT_EQ(fused_gates[1].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[1].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[1].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[2]->qubits[1], 3);
  EXPECT_EQ(fused_gates[1].gates[3]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[3]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[4]->kind, kGateT);
  EXPECT_EQ(fused_gates[1].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[4]->qubits[0], 3);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].num_qubits, 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 9);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 4);
  EXPECT_EQ(fused_gates[2].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 5);
  EXPECT_EQ(fused_gates[2].gates[3]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[3]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[4]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[4]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[5]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[5]->time, 6);
  EXPECT_EQ(fused_gates[2].gates[5]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[5]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[6]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[6]->time, 7);
  EXPECT_EQ(fused_gates[2].gates[6]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[6]->qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates[7]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[7]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[7]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[7]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[8]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[8]->time, 8);
  EXPECT_EQ(fused_gates[2].gates[8]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[8]->qubits[0], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].time, 9);
  EXPECT_EQ(fused_gates[3].num_qubits, 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates.size(), 4);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateY);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[3].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 9);
  EXPECT_EQ(fused_gates[3].gates[1]->num_qubits, 2);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->qubits[1], 1);
  EXPECT_EQ(fused_gates[3].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[3]->kind, kGateHd);
  EXPECT_EQ(fused_gates[3].gates[3]->time, 10);
  EXPECT_EQ(fused_gates[3].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[3]->qubits[0], 1);

  EXPECT_EQ(fused_gates[4].kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].time, 9);
  EXPECT_EQ(fused_gates[4].num_qubits, 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 4);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 3);
  EXPECT_EQ(fused_gates[4].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 3);
  EXPECT_EQ(fused_gates[4].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[4].gates[1]->time, 9);
  EXPECT_EQ(fused_gates[4].gates[1]->num_qubits, 2);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[1]->qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates[2]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[2]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[4].gates[2]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[3]->kind, kGateHd);
  EXPECT_EQ(fused_gates[4].gates[3]->time, 10);
  EXPECT_EQ(fused_gates[4].gates[3]->num_qubits, 1);
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
  auto fused_gates = Fuser::FuseGates(circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 2);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 5);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates[3]->kind, kGateT);
  EXPECT_EQ(fused_gates[0].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[3]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[3]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[4]->kind, kGateX);
  EXPECT_EQ(fused_gates[0].gates[4]->time, 2);
  EXPECT_EQ(fused_gates[0].gates[4]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[4]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].num_qubits, 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[1]->num_qubits, 1);
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
  auto fused_gates = Fuser::FuseGates(
      circuit.num_qubits, circuit.gates, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 4);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 3);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].num_qubits, 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 3);
  EXPECT_EQ(fused_gates[2].num_qubits, 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 1);
  EXPECT_EQ(fused_gates[2].qubits[1], 2);
  EXPECT_EQ(fused_gates[2].gates.size(), 3);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 2);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 3);
  EXPECT_EQ(fused_gates[2].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[1], 2);

  EXPECT_EQ(fused_gates[3].kind, kGateT);
  EXPECT_EQ(fused_gates[3].time, 2);
  EXPECT_EQ(fused_gates[3].num_qubits, 1);
  EXPECT_EQ(fused_gates[3].qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates.size(), 2);
  EXPECT_EQ(fused_gates[3].gates[0]->kind, kGateT);
  EXPECT_EQ(fused_gates[3].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[3].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[3].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[3].gates[1]->kind, kGateX);
  EXPECT_EQ(fused_gates[3].gates[1]->time, 4);
  EXPECT_EQ(fused_gates[3].gates[1]->num_qubits, 1);
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
  auto fused_gates = Fuser::FuseGates(circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 3);

  EXPECT_EQ(fused_gates[0].kind, kGateHd);
  EXPECT_EQ(fused_gates[0].time, 0);
  EXPECT_EQ(fused_gates[0].num_qubits, 1);
  EXPECT_EQ(fused_gates[0].qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].num_qubits, 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 2);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[1].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[1].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[1]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].time, 1);
  EXPECT_EQ(fused_gates[2].num_qubits, 2);
  EXPECT_EQ(fused_gates[2].qubits[0], 0);
  EXPECT_EQ(fused_gates[2].qubits[1], 1);
  EXPECT_EQ(fused_gates[2].gates.size(), 4);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[1]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[2].gates[1]->time, 1);
  EXPECT_EQ(fused_gates[2].gates[1]->num_qubits, 2);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[1]->qubits[1], 1);
  EXPECT_EQ(fused_gates[2].gates[2]->kind, kGateT);
  EXPECT_EQ(fused_gates[2].gates[2]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[2]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[2].gates[3]->kind, kGateX);
  EXPECT_EQ(fused_gates[2].gates[3]->time, 2);
  EXPECT_EQ(fused_gates[2].gates[3]->num_qubits, 1);
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

  using Fuser = BasicGateFuser<IO, GateQSim<float>>;
  auto fused_gates = Fuser::FuseGates(circuit.num_qubits, circuit.gates);

  EXPECT_EQ(fused_gates.size(), 11);

  EXPECT_EQ(fused_gates[0].kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].time, 1);
  EXPECT_EQ(fused_gates[0].num_qubits, 2);
  EXPECT_EQ(fused_gates[0].qubits[0], 0);
  EXPECT_EQ(fused_gates[0].qubits[1], 1);
  EXPECT_EQ(fused_gates[0].gates.size(), 3);
  EXPECT_EQ(fused_gates[0].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[1]->kind, kGateHd);
  EXPECT_EQ(fused_gates[0].gates[1]->time, 0);
  EXPECT_EQ(fused_gates[0].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[0].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[0].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[0].gates[2]->time, 1);
  EXPECT_EQ(fused_gates[0].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[0].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[1].kind, kGateHd);
  EXPECT_EQ(fused_gates[1].time, 0);
  EXPECT_EQ(fused_gates[1].num_qubits, 1);
  EXPECT_EQ(fused_gates[1].qubits[0], 2);
  EXPECT_EQ(fused_gates[1].gates.size(), 1);
  EXPECT_EQ(fused_gates[1].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[1].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[1].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[1].gates[0]->qubits[0], 2);

  EXPECT_EQ(fused_gates[2].kind, kGateHd);
  EXPECT_EQ(fused_gates[2].time, 0);
  EXPECT_EQ(fused_gates[2].num_qubits, 1);
  EXPECT_EQ(fused_gates[2].qubits[0], 3);
  EXPECT_EQ(fused_gates[2].gates.size(), 1);
  EXPECT_EQ(fused_gates[2].gates[0]->kind, kGateHd);
  EXPECT_EQ(fused_gates[2].gates[0]->time, 0);
  EXPECT_EQ(fused_gates[2].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[2].gates[0]->qubits[0], 3);

  EXPECT_EQ(fused_gates[3].kind, kMeasurement);
  EXPECT_EQ(fused_gates[3].time, 1);
  EXPECT_EQ(fused_gates[3].num_qubits, 2);
  EXPECT_EQ(fused_gates[3].qubits[0], 2);
  EXPECT_EQ(fused_gates[3].qubits[1], 3);
  EXPECT_EQ(fused_gates[3].gates.size(), 0);

  EXPECT_EQ(fused_gates[4].kind, kGateIS);
  EXPECT_EQ(fused_gates[4].time, 2);
  EXPECT_EQ(fused_gates[4].num_qubits, 2);
  EXPECT_EQ(fused_gates[4].qubits[0], 2);
  EXPECT_EQ(fused_gates[4].qubits[1], 3);
  EXPECT_EQ(fused_gates[4].gates.size(), 1);
  EXPECT_EQ(fused_gates[4].gates[0]->kind, kGateIS);
  EXPECT_EQ(fused_gates[4].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[4].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[4].gates[0]->qubits[1], 3);

  EXPECT_EQ(fused_gates[5].kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].time, 3);
  EXPECT_EQ(fused_gates[5].num_qubits, 2);
  EXPECT_EQ(fused_gates[5].qubits[0], 0);
  EXPECT_EQ(fused_gates[5].qubits[1], 1);
  EXPECT_EQ(fused_gates[5].gates.size(), 3);
  EXPECT_EQ(fused_gates[5].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[5].gates[0]->time, 2);
  EXPECT_EQ(fused_gates[5].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[5].gates[0]->qubits[0], 0);
  EXPECT_EQ(fused_gates[5].gates[1]->kind, kGateY);
  EXPECT_EQ(fused_gates[5].gates[1]->time, 2);
  EXPECT_EQ(fused_gates[5].gates[1]->num_qubits, 1);
  EXPECT_EQ(fused_gates[5].gates[1]->qubits[0], 1);
  EXPECT_EQ(fused_gates[5].gates[2]->kind, kGateCZ);
  EXPECT_EQ(fused_gates[5].gates[2]->time, 3);
  EXPECT_EQ(fused_gates[5].gates[2]->num_qubits, 2);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits[0], 0);
  EXPECT_EQ(fused_gates[5].gates[2]->qubits[1], 1);

  EXPECT_EQ(fused_gates[6].kind, kMeasurement);
  EXPECT_EQ(fused_gates[6].time, 3);
  EXPECT_EQ(fused_gates[6].num_qubits, 2);
  EXPECT_EQ(fused_gates[6].qubits[0], 2);
  EXPECT_EQ(fused_gates[6].qubits[1], 3);
  EXPECT_EQ(fused_gates[6].gates.size(), 0);

  EXPECT_EQ(fused_gates[7].kind, kGateIS);
  EXPECT_EQ(fused_gates[7].time, 4);
  EXPECT_EQ(fused_gates[7].num_qubits, 2);
  EXPECT_EQ(fused_gates[7].qubits[0], 2);
  EXPECT_EQ(fused_gates[7].qubits[1], 3);
  EXPECT_EQ(fused_gates[7].gates.size(), 1);
  EXPECT_EQ(fused_gates[7].gates[0]->kind, kGateIS);
  EXPECT_EQ(fused_gates[7].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[7].gates[0]->num_qubits, 2);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits[0], 2);
  EXPECT_EQ(fused_gates[7].gates[0]->qubits[1], 3);

  EXPECT_EQ(fused_gates[8].kind, kGateX);
  EXPECT_EQ(fused_gates[8].time, 4);
  EXPECT_EQ(fused_gates[8].num_qubits, 1);
  EXPECT_EQ(fused_gates[8].qubits[0], 0);
  EXPECT_EQ(fused_gates[8].gates.size(), 1);
  EXPECT_EQ(fused_gates[8].gates[0]->kind, kGateX);
  EXPECT_EQ(fused_gates[8].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[8].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[8].gates[0]->qubits[0], 0);

  EXPECT_EQ(fused_gates[9].kind, kGateY);
  EXPECT_EQ(fused_gates[9].time, 4);
  EXPECT_EQ(fused_gates[9].num_qubits, 1);
  EXPECT_EQ(fused_gates[9].qubits[0], 1);
  EXPECT_EQ(fused_gates[9].gates.size(), 1);
  EXPECT_EQ(fused_gates[9].gates[0]->kind, kGateY);
  EXPECT_EQ(fused_gates[9].gates[0]->time, 4);
  EXPECT_EQ(fused_gates[9].gates[0]->num_qubits, 1);
  EXPECT_EQ(fused_gates[9].gates[0]->qubits[0], 1);

  EXPECT_EQ(fused_gates[10].kind, kMeasurement);
  EXPECT_EQ(fused_gates[10].time, 5);
  EXPECT_EQ(fused_gates[10].num_qubits, 4);
  EXPECT_EQ(fused_gates[10].qubits[0], 2);
  EXPECT_EQ(fused_gates[10].qubits[1], 3);
  EXPECT_EQ(fused_gates[10].qubits[2], 0);
  EXPECT_EQ(fused_gates[10].qubits[3], 1);
  EXPECT_EQ(fused_gates[10].gates.size(), 0);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
