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
#include <vector>

#include "fuser_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_basic.h"
#include "../lib/gate.h"
#include "../lib/operation.h"
#include "../lib/operation_base.h"

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
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string1);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 27);

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, circuit.ops);

  EXPECT_EQ(fused_gates.size(), 5);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 6);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);
  const auto* gate03 = OpGetAlternative<Gate>(fgate0->gates[3]);
  ASSERT_NE(gate03, nullptr);
  EXPECT_EQ(gate03->kind, kGateT);
  EXPECT_EQ(gate03->time, 2);
  EXPECT_EQ(gate03->qubits.size(), 1);
  EXPECT_EQ(gate03->qubits[0], 0);
  const auto* gate04 = OpGetAlternative<Gate>(fgate0->gates[4]);
  ASSERT_NE(gate04, nullptr);
  EXPECT_EQ(gate04->kind, kGateY);
  EXPECT_EQ(gate04->time, 3);
  EXPECT_EQ(gate04->qubits.size(), 1);
  EXPECT_EQ(gate04->qubits[0], 0);
  const auto* gate05 = OpGetAlternative<Gate>(fgate0->gates[5]);
  ASSERT_NE(gate05, nullptr);
  EXPECT_EQ(gate05->kind, kGateX);
  EXPECT_EQ(gate05->time, 2);
  EXPECT_EQ(gate05->qubits.size(), 1);
  EXPECT_EQ(gate05->qubits[0], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateCZ);
  EXPECT_EQ(fgate1->time, 1);
  EXPECT_EQ(fgate1->qubits.size(), 2);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->qubits[1], 3);
  EXPECT_EQ(fgate1->gates.size(), 6);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateHd);
  EXPECT_EQ(gate11->time, 0);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 3);
  const auto* gate12 = OpGetAlternative<Gate>(fgate1->gates[2]);
  ASSERT_NE(gate12, nullptr);
  EXPECT_EQ(gate12->kind, kGateCZ);
  EXPECT_EQ(gate12->time, 1);
  EXPECT_EQ(gate12->qubits.size(), 2);
  EXPECT_EQ(gate12->qubits[0], 2);
  EXPECT_EQ(gate12->qubits[1], 3);
  const auto* gate13 = OpGetAlternative<Gate>(fgate1->gates[3]);
  ASSERT_NE(gate13, nullptr);
  EXPECT_EQ(gate13->kind, kGateY);
  EXPECT_EQ(gate13->time, 2);
  EXPECT_EQ(gate13->qubits.size(), 1);
  EXPECT_EQ(gate13->qubits[0], 2);
  const auto* gate14 = OpGetAlternative<Gate>(fgate1->gates[4]);
  ASSERT_NE(gate14, nullptr);
  EXPECT_EQ(gate14->kind, kGateT);
  EXPECT_EQ(gate14->time, 2);
  EXPECT_EQ(gate14->qubits.size(), 1);
  EXPECT_EQ(gate14->qubits[0], 3);
  const auto* gate15 = OpGetAlternative<Gate>(fgate1->gates[5]);
  ASSERT_NE(gate15, nullptr);
  EXPECT_EQ(gate15->kind, kGateX);
  EXPECT_EQ(gate15->time, 3);
  EXPECT_EQ(gate15->qubits.size(), 1);
  EXPECT_EQ(gate15->qubits[0], 3);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateCZ);
  EXPECT_EQ(fgate2->time, 3);
  EXPECT_EQ(fgate2->qubits.size(), 2);
  EXPECT_EQ(fgate2->qubits[0], 1);
  EXPECT_EQ(fgate2->qubits[1], 2);
  EXPECT_EQ(fgate2->gates.size(), 9);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateCZ);
  EXPECT_EQ(gate20->time, 3);
  EXPECT_EQ(gate20->qubits.size(), 2);
  EXPECT_EQ(gate20->qubits[0], 1);
  EXPECT_EQ(gate20->qubits[1], 2);
  const auto* gate21 = OpGetAlternative<Gate>(fgate2->gates[1]);
  ASSERT_NE(gate21, nullptr);
  EXPECT_EQ(gate21->kind, kGateT);
  EXPECT_EQ(gate21->time, 4);
  EXPECT_EQ(gate21->qubits.size(), 1);
  EXPECT_EQ(gate21->qubits[0], 1);
  const auto* gate22 = OpGetAlternative<Gate>(fgate2->gates[2]);
  ASSERT_NE(gate22, nullptr);
  EXPECT_EQ(gate22->kind, kGateT);
  EXPECT_EQ(gate22->time, 4);
  EXPECT_EQ(gate22->qubits.size(), 1);
  EXPECT_EQ(gate22->qubits[0], 2);
  const auto* gate23 = OpGetAlternative<Gate>(fgate2->gates[3]);
  ASSERT_NE(gate23, nullptr);
  EXPECT_EQ(gate23->kind, kGateCZ);
  EXPECT_EQ(gate23->time, 5);
  EXPECT_EQ(gate23->qubits.size(), 2);
  EXPECT_EQ(gate23->qubits[0], 1);
  EXPECT_EQ(gate23->qubits[1], 2);
  const auto* gate24 = OpGetAlternative<Gate>(fgate2->gates[4]);
  ASSERT_NE(gate24, nullptr);
  EXPECT_EQ(gate24->kind, kGateX);
  EXPECT_EQ(gate24->time, 6);
  EXPECT_EQ(gate24->qubits.size(), 1);
  EXPECT_EQ(gate24->qubits[0], 1);
  const auto* gate25 = OpGetAlternative<Gate>(fgate2->gates[5]);
  ASSERT_NE(gate25, nullptr);
  EXPECT_EQ(gate25->kind, kGateY);
  EXPECT_EQ(gate25->time, 6);
  EXPECT_EQ(gate25->qubits.size(), 1);
  EXPECT_EQ(gate25->qubits[0], 2);
  const auto* gate26 = OpGetAlternative<Gate>(fgate2->gates[6]);
  ASSERT_NE(gate26, nullptr);
  EXPECT_EQ(gate26->kind, kGateCZ);
  EXPECT_EQ(gate26->time, 7);
  EXPECT_EQ(gate26->qubits.size(), 2);
  EXPECT_EQ(gate26->qubits[0], 1);
  EXPECT_EQ(gate26->qubits[1], 2);
  const auto* gate27 = OpGetAlternative<Gate>(fgate2->gates[7]);
  ASSERT_NE(gate27, nullptr);
  EXPECT_EQ(gate27->kind, kGateT);
  EXPECT_EQ(gate27->time, 8);
  EXPECT_EQ(gate27->qubits.size(), 1);
  EXPECT_EQ(gate27->qubits[0], 1);
  const auto* gate28 = OpGetAlternative<Gate>(fgate2->gates[8]);
  ASSERT_NE(gate28, nullptr);
  EXPECT_EQ(gate28->kind, kGateT);
  EXPECT_EQ(gate28->time, 8);
  EXPECT_EQ(gate28->qubits.size(), 1);
  EXPECT_EQ(gate28->qubits[0], 2);

  const auto* fgate3 = OpGetAlternative<FusedGate>(fused_gates[3]);
  ASSERT_NE(fgate3, nullptr);
  EXPECT_EQ(fgate3->kind, kGateCZ);
  EXPECT_EQ(fgate3->time, 9);
  EXPECT_EQ(fgate3->qubits.size(), 2);
  EXPECT_EQ(fgate3->qubits[0], 0);
  EXPECT_EQ(fgate3->qubits[1], 1);
  EXPECT_EQ(fgate3->gates.size(), 3);
  const auto* gate30 = OpGetAlternative<Gate>(fgate3->gates[0]);
  ASSERT_NE(gate30, nullptr);
  EXPECT_EQ(gate30->kind, kGateCZ);
  EXPECT_EQ(gate30->time, 9);
  EXPECT_EQ(gate30->qubits.size(), 2);
  EXPECT_EQ(gate30->qubits[0], 0);
  EXPECT_EQ(gate30->qubits[1], 1);
  const auto* gate31 = OpGetAlternative<Gate>(fgate3->gates[1]);
  ASSERT_NE(gate31, nullptr);
  EXPECT_EQ(gate31->kind, kGateHd);
  EXPECT_EQ(gate31->time, 10);
  EXPECT_EQ(gate31->qubits.size(), 1);
  EXPECT_EQ(gate31->qubits[0], 0);
  const auto* gate32 = OpGetAlternative<Gate>(fgate3->gates[2]);
  ASSERT_NE(gate32, nullptr);
  EXPECT_EQ(gate32->kind, kGateHd);
  EXPECT_EQ(gate32->time, 10);
  EXPECT_EQ(gate32->qubits.size(), 1);
  EXPECT_EQ(gate32->qubits[0], 1);

  const auto* fgate4 = OpGetAlternative<FusedGate>(fused_gates[4]);
  ASSERT_NE(fgate4, nullptr);
  EXPECT_EQ(fgate4->kind, kGateCZ);
  EXPECT_EQ(fgate4->time, 9);
  EXPECT_EQ(fgate4->qubits.size(), 2);
  EXPECT_EQ(fgate4->qubits[0], 2);
  EXPECT_EQ(fgate4->qubits[1], 3);
  EXPECT_EQ(fgate4->gates.size(), 3);
  const auto* gate40 = OpGetAlternative<Gate>(fgate4->gates[0]);
  ASSERT_NE(gate40, nullptr);
  EXPECT_EQ(gate40->kind, kGateCZ);
  EXPECT_EQ(gate40->time, 9);
  EXPECT_EQ(gate40->qubits.size(), 2);
  EXPECT_EQ(gate40->qubits[0], 2);
  EXPECT_EQ(gate40->qubits[1], 3);
  const auto* gate41 = OpGetAlternative<Gate>(fgate4->gates[1]);
  ASSERT_NE(gate41, nullptr);
  EXPECT_EQ(gate41->kind, kGateHd);
  EXPECT_EQ(gate41->time, 10);
  EXPECT_EQ(gate41->qubits.size(), 1);
  EXPECT_EQ(gate41->qubits[0], 2);
  const auto* gate42 = OpGetAlternative<Gate>(fgate4->gates[2]);
  ASSERT_NE(gate42, nullptr);
  EXPECT_EQ(gate42->kind, kGateHd);
  EXPECT_EQ(gate42->time, 10);
  EXPECT_EQ(gate42->qubits.size(), 1);
  EXPECT_EQ(gate42->qubits[0], 3);
}

TEST(FuserBasicTest, TimesToSplitAt1) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string1);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 27);

  std::vector<unsigned> times_to_split_at{3, 8, 10};

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.ops, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 6);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 6);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);
  const auto* gate03 = OpGetAlternative<Gate>(fgate0->gates[3]);
  ASSERT_NE(gate03, nullptr);
  EXPECT_EQ(gate03->kind, kGateT);
  EXPECT_EQ(gate03->time, 2);
  EXPECT_EQ(gate03->qubits.size(), 1);
  EXPECT_EQ(gate03->qubits[0], 0);
  const auto* gate04 = OpGetAlternative<Gate>(fgate0->gates[4]);
  ASSERT_NE(gate04, nullptr);
  EXPECT_EQ(gate04->kind, kGateY);
  EXPECT_EQ(gate04->time, 3);
  EXPECT_EQ(gate04->qubits.size(), 1);
  EXPECT_EQ(gate04->qubits[0], 0);
  const auto* gate05 = OpGetAlternative<Gate>(fgate0->gates[5]);
  ASSERT_NE(gate05, nullptr);
  EXPECT_EQ(gate05->kind, kGateX);
  EXPECT_EQ(gate05->time, 2);
  EXPECT_EQ(gate05->qubits.size(), 1);
  EXPECT_EQ(gate05->qubits[0], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateCZ);
  EXPECT_EQ(fgate1->time, 1);
  EXPECT_EQ(fgate1->qubits.size(), 2);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->qubits[1], 3);
  EXPECT_EQ(fgate1->gates.size(), 6);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateHd);
  EXPECT_EQ(gate11->time, 0);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 3);
  const auto* gate12 = OpGetAlternative<Gate>(fgate1->gates[2]);
  ASSERT_NE(gate12, nullptr);
  EXPECT_EQ(gate12->kind, kGateCZ);
  EXPECT_EQ(gate12->time, 1);
  EXPECT_EQ(gate12->qubits.size(), 2);
  EXPECT_EQ(gate12->qubits[0], 2);
  EXPECT_EQ(gate12->qubits[1], 3);
  const auto* gate13 = OpGetAlternative<Gate>(fgate1->gates[3]);
  ASSERT_NE(gate13, nullptr);
  EXPECT_EQ(gate13->kind, kGateY);
  EXPECT_EQ(gate13->time, 2);
  EXPECT_EQ(gate13->qubits.size(), 1);
  EXPECT_EQ(gate13->qubits[0], 2);
  const auto* gate14 = OpGetAlternative<Gate>(fgate1->gates[4]);
  ASSERT_NE(gate14, nullptr);
  EXPECT_EQ(gate14->kind, kGateT);
  EXPECT_EQ(gate14->time, 2);
  EXPECT_EQ(gate14->qubits.size(), 1);
  EXPECT_EQ(gate14->qubits[0], 3);
  const auto* gate15 = OpGetAlternative<Gate>(fgate1->gates[5]);
  ASSERT_NE(gate15, nullptr);
  EXPECT_EQ(gate15->kind, kGateX);
  EXPECT_EQ(gate15->time, 3);
  EXPECT_EQ(gate15->qubits.size(), 1);
  EXPECT_EQ(gate15->qubits[0], 3);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateCZ);
  EXPECT_EQ(fgate2->time, 3);
  EXPECT_EQ(fgate2->qubits.size(), 2);
  EXPECT_EQ(fgate2->qubits[0], 1);
  EXPECT_EQ(fgate2->qubits[1], 2);
  EXPECT_EQ(fgate2->gates.size(), 1);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateCZ);
  EXPECT_EQ(gate20->time, 3);
  EXPECT_EQ(gate20->qubits.size(), 2);
  EXPECT_EQ(gate20->qubits[0], 1);
  EXPECT_EQ(gate20->qubits[1], 2);

  const auto* fgate3 = OpGetAlternative<FusedGate>(fused_gates[3]);
  ASSERT_NE(fgate3, nullptr);
  EXPECT_EQ(fgate3->kind, kGateCZ);
  EXPECT_EQ(fgate3->time, 5);
  EXPECT_EQ(fgate3->qubits.size(), 2);
  EXPECT_EQ(fgate3->qubits[0], 1);
  EXPECT_EQ(fgate3->qubits[1], 2);
  EXPECT_EQ(fgate3->gates.size(), 8);
  const auto* gate30 = OpGetAlternative<Gate>(fgate3->gates[0]);
  ASSERT_NE(gate30, nullptr);
  EXPECT_EQ(gate30->kind, kGateT);
  EXPECT_EQ(gate30->time, 4);
  EXPECT_EQ(gate30->qubits.size(), 1);
  EXPECT_EQ(gate30->qubits[0], 1);
  const auto* gate31 = OpGetAlternative<Gate>(fgate3->gates[1]);
  ASSERT_NE(gate31, nullptr);
  EXPECT_EQ(gate31->kind, kGateT);
  EXPECT_EQ(gate31->time, 4);
  EXPECT_EQ(gate31->qubits.size(), 1);
  EXPECT_EQ(gate31->qubits[0], 2);
  const auto* gate32 = OpGetAlternative<Gate>(fgate3->gates[2]);
  ASSERT_NE(gate32, nullptr);
  EXPECT_EQ(gate32->kind, kGateCZ);
  EXPECT_EQ(gate32->time, 5);
  EXPECT_EQ(gate32->qubits.size(), 2);
  EXPECT_EQ(gate32->qubits[0], 1);
  EXPECT_EQ(gate32->qubits[1], 2);
  const auto* gate33 = OpGetAlternative<Gate>(fgate3->gates[3]);
  ASSERT_NE(gate33, nullptr);
  EXPECT_EQ(gate33->kind, kGateX);
  EXPECT_EQ(gate33->time, 6);
  EXPECT_EQ(gate33->qubits.size(), 1);
  EXPECT_EQ(gate33->qubits[0], 1);
  const auto* gate34 = OpGetAlternative<Gate>(fgate3->gates[4]);
  ASSERT_NE(gate34, nullptr);
  EXPECT_EQ(gate34->kind, kGateY);
  EXPECT_EQ(gate34->time, 6);
  EXPECT_EQ(gate34->qubits.size(), 1);
  EXPECT_EQ(gate34->qubits[0], 2);
  const auto* gate35 = OpGetAlternative<Gate>(fgate3->gates[5]);
  ASSERT_NE(gate35, nullptr);
  EXPECT_EQ(gate35->kind, kGateCZ);
  EXPECT_EQ(gate35->time, 7);
  EXPECT_EQ(gate35->qubits.size(), 2);
  EXPECT_EQ(gate35->qubits[0], 1);
  EXPECT_EQ(gate35->qubits[1], 2);
  const auto* gate36 = OpGetAlternative<Gate>(fgate3->gates[6]);
  ASSERT_NE(gate36, nullptr);
  EXPECT_EQ(gate36->kind, kGateT);
  EXPECT_EQ(gate36->time, 8);
  EXPECT_EQ(gate36->qubits.size(), 1);
  EXPECT_EQ(gate36->qubits[0], 1);
  const auto* gate37 = OpGetAlternative<Gate>(fgate3->gates[7]);
  ASSERT_NE(gate37, nullptr);
  EXPECT_EQ(gate37->kind, kGateT);
  EXPECT_EQ(gate37->time, 8);
  EXPECT_EQ(gate37->qubits.size(), 1);
  EXPECT_EQ(gate37->qubits[0], 2);

  const auto* fgate4 = OpGetAlternative<FusedGate>(fused_gates[4]);
  ASSERT_NE(fgate4, nullptr);
  EXPECT_EQ(fgate4->kind, kGateCZ);
  EXPECT_EQ(fgate4->time, 9);
  EXPECT_EQ(fgate4->qubits.size(), 2);
  EXPECT_EQ(fgate4->qubits[0], 0);
  EXPECT_EQ(fgate4->qubits[1], 1);
  EXPECT_EQ(fgate4->gates.size(), 3);
  const auto* gate40 = OpGetAlternative<Gate>(fgate4->gates[0]);
  ASSERT_NE(gate40, nullptr);
  EXPECT_EQ(gate40->kind, kGateCZ);
  EXPECT_EQ(gate40->time, 9);
  EXPECT_EQ(gate40->qubits.size(), 2);
  EXPECT_EQ(gate40->qubits[0], 0);
  EXPECT_EQ(gate40->qubits[1], 1);
  const auto* gate41 = OpGetAlternative<Gate>(fgate4->gates[1]);
  ASSERT_NE(gate41, nullptr);
  EXPECT_EQ(gate41->kind, kGateHd);
  EXPECT_EQ(gate41->time, 10);
  EXPECT_EQ(gate41->qubits.size(), 1);
  EXPECT_EQ(gate41->qubits[0], 0);
  const auto* gate42 = OpGetAlternative<Gate>(fgate4->gates[2]);
  ASSERT_NE(gate42, nullptr);
  EXPECT_EQ(gate42->kind, kGateHd);
  EXPECT_EQ(gate42->time, 10);
  EXPECT_EQ(gate42->qubits.size(), 1);
  EXPECT_EQ(gate42->qubits[0], 1);

  const auto* fgate5 = OpGetAlternative<FusedGate>(fused_gates[5]);
  ASSERT_NE(fgate5, nullptr);
  EXPECT_EQ(fgate5->kind, kGateCZ);
  EXPECT_EQ(fgate5->time, 9);
  EXPECT_EQ(fgate5->qubits.size(), 2);
  EXPECT_EQ(fgate5->qubits[0], 2);
  EXPECT_EQ(fgate5->qubits[1], 3);
  EXPECT_EQ(fgate5->gates.size(), 3);
  const auto* gate50 = OpGetAlternative<Gate>(fgate5->gates[0]);
  ASSERT_NE(gate50, nullptr);
  EXPECT_EQ(gate50->kind, kGateCZ);
  EXPECT_EQ(gate50->time, 9);
  EXPECT_EQ(gate50->qubits.size(), 2);
  EXPECT_EQ(gate50->qubits[0], 2);
  EXPECT_EQ(gate50->qubits[1], 3);
  const auto* gate51 = OpGetAlternative<Gate>(fgate5->gates[1]);
  ASSERT_NE(gate51, nullptr);
  EXPECT_EQ(gate51->kind, kGateHd);
  EXPECT_EQ(gate51->time, 10);
  EXPECT_EQ(gate51->qubits.size(), 1);
  EXPECT_EQ(gate51->qubits[0], 2);
  const auto* gate52 = OpGetAlternative<Gate>(fgate5->gates[2]);
  ASSERT_NE(gate52, nullptr);
  EXPECT_EQ(gate52->kind, kGateHd);
  EXPECT_EQ(gate52->time, 10);
  EXPECT_EQ(gate52->qubits.size(), 1);
  EXPECT_EQ(gate52->qubits[0], 3);
}

TEST(FuserBasicTest, TimesToSplitAt2) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string1);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 27);

  std::vector<unsigned> times_to_split_at{2, 10};

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.ops, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 5);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 5);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);
  const auto* gate03 = OpGetAlternative<Gate>(fgate0->gates[3]);
  ASSERT_NE(gate03, nullptr);
  EXPECT_EQ(gate03->kind, kGateT);
  EXPECT_EQ(gate03->time, 2);
  EXPECT_EQ(gate03->qubits.size(), 1);
  EXPECT_EQ(gate03->qubits[0], 0);
  const auto* gate04 = OpGetAlternative<Gate>(fgate0->gates[4]);
  ASSERT_NE(gate04, nullptr);
  EXPECT_EQ(gate04->kind, kGateX);
  EXPECT_EQ(gate04->time, 2);
  EXPECT_EQ(gate04->qubits.size(), 1);
  EXPECT_EQ(gate04->qubits[0], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateCZ);
  EXPECT_EQ(fgate1->time, 1);
  EXPECT_EQ(fgate1->qubits.size(), 2);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->qubits[1], 3);
  EXPECT_EQ(fgate1->gates.size(), 5);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateHd);
  EXPECT_EQ(gate11->time, 0);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 3);
  const auto* gate12 = OpGetAlternative<Gate>(fgate1->gates[2]);
  ASSERT_NE(gate12, nullptr);
  EXPECT_EQ(gate12->kind, kGateCZ);
  EXPECT_EQ(gate12->time, 1);
  EXPECT_EQ(gate12->qubits.size(), 2);
  EXPECT_EQ(gate12->qubits[0], 2);
  EXPECT_EQ(gate12->qubits[1], 3);
  const auto* gate13 = OpGetAlternative<Gate>(fgate1->gates[3]);
  ASSERT_NE(gate13, nullptr);
  EXPECT_EQ(gate13->kind, kGateY);
  EXPECT_EQ(gate13->time, 2);
  EXPECT_EQ(gate13->qubits.size(), 1);
  EXPECT_EQ(gate13->qubits[0], 2);
  const auto* gate14 = OpGetAlternative<Gate>(fgate1->gates[4]);
  ASSERT_NE(gate14, nullptr);
  EXPECT_EQ(gate14->kind, kGateT);
  EXPECT_EQ(gate14->time, 2);
  EXPECT_EQ(gate14->qubits.size(), 1);
  EXPECT_EQ(gate14->qubits[0], 3);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateCZ);
  EXPECT_EQ(fgate2->time, 3);
  EXPECT_EQ(fgate2->qubits.size(), 2);
  EXPECT_EQ(fgate2->qubits[0], 1);
  EXPECT_EQ(fgate2->qubits[1], 2);
  EXPECT_EQ(fgate2->gates.size(), 9);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateCZ);
  EXPECT_EQ(gate20->time, 3);
  EXPECT_EQ(gate20->qubits.size(), 2);
  EXPECT_EQ(gate20->qubits[0], 1);
  EXPECT_EQ(gate20->qubits[1], 2);
  const auto* gate21 = OpGetAlternative<Gate>(fgate2->gates[1]);
  ASSERT_NE(gate21, nullptr);
  EXPECT_EQ(gate21->kind, kGateT);
  EXPECT_EQ(gate21->time, 4);
  EXPECT_EQ(gate21->qubits.size(), 1);
  EXPECT_EQ(gate21->qubits[0], 1);
  const auto* gate22 = OpGetAlternative<Gate>(fgate2->gates[2]);
  ASSERT_NE(gate22, nullptr);
  EXPECT_EQ(gate22->kind, kGateT);
  EXPECT_EQ(gate22->time, 4);
  EXPECT_EQ(gate22->qubits.size(), 1);
  EXPECT_EQ(gate22->qubits[0], 2);
  const auto* gate23 = OpGetAlternative<Gate>(fgate2->gates[3]);
  ASSERT_NE(gate23, nullptr);
  EXPECT_EQ(gate23->kind, kGateCZ);
  EXPECT_EQ(gate23->time, 5);
  EXPECT_EQ(gate23->qubits.size(), 2);
  EXPECT_EQ(gate23->qubits[0], 1);
  EXPECT_EQ(gate23->qubits[1], 2);
  const auto* gate24 = OpGetAlternative<Gate>(fgate2->gates[4]);
  ASSERT_NE(gate24, nullptr);
  EXPECT_EQ(gate24->kind, kGateX);
  EXPECT_EQ(gate24->time, 6);
  EXPECT_EQ(gate24->qubits.size(), 1);
  EXPECT_EQ(gate24->qubits[0], 1);
  const auto* gate25 = OpGetAlternative<Gate>(fgate2->gates[5]);
  ASSERT_NE(gate25, nullptr);
  EXPECT_EQ(gate25->kind, kGateY);
  EXPECT_EQ(gate25->time, 6);
  EXPECT_EQ(gate25->qubits.size(), 1);
  EXPECT_EQ(gate25->qubits[0], 2);
  const auto* gate26 = OpGetAlternative<Gate>(fgate2->gates[6]);
  ASSERT_NE(gate26, nullptr);
  EXPECT_EQ(gate26->kind, kGateCZ);
  EXPECT_EQ(gate26->time, 7);
  EXPECT_EQ(gate26->qubits.size(), 2);
  EXPECT_EQ(gate26->qubits[0], 1);
  EXPECT_EQ(gate26->qubits[1], 2);
  const auto* gate27 = OpGetAlternative<Gate>(fgate2->gates[7]);
  ASSERT_NE(gate27, nullptr);
  EXPECT_EQ(gate27->kind, kGateT);
  EXPECT_EQ(gate27->time, 8);
  EXPECT_EQ(gate27->qubits.size(), 1);
  EXPECT_EQ(gate27->qubits[0], 1);
  const auto* gate28 = OpGetAlternative<Gate>(fgate2->gates[8]);
  ASSERT_NE(gate28, nullptr);
  EXPECT_EQ(gate28->kind, kGateT);
  EXPECT_EQ(gate28->time, 8);
  EXPECT_EQ(gate28->qubits.size(), 1);
  EXPECT_EQ(gate28->qubits[0], 2);

  const auto* fgate3 = OpGetAlternative<FusedGate>(fused_gates[3]);
  ASSERT_NE(fgate3, nullptr);
  EXPECT_EQ(fgate3->kind, kGateCZ);
  EXPECT_EQ(fgate3->time, 9);
  EXPECT_EQ(fgate3->qubits.size(), 2);
  EXPECT_EQ(fgate3->qubits[0], 0);
  EXPECT_EQ(fgate3->qubits[1], 1);
  EXPECT_EQ(fgate3->gates.size(), 4);
  const auto* gate30 = OpGetAlternative<Gate>(fgate3->gates[0]);
  ASSERT_NE(gate30, nullptr);
  EXPECT_EQ(gate30->kind, kGateY);
  EXPECT_EQ(gate30->time, 3);
  EXPECT_EQ(gate30->qubits.size(), 1);
  EXPECT_EQ(gate30->qubits[0], 0);
  const auto* gate31 = OpGetAlternative<Gate>(fgate3->gates[1]);
  ASSERT_NE(gate31, nullptr);
  EXPECT_EQ(gate31->kind, kGateCZ);
  EXPECT_EQ(gate31->time, 9);
  EXPECT_EQ(gate31->qubits.size(), 2);
  EXPECT_EQ(gate31->qubits[0], 0);
  EXPECT_EQ(gate31->qubits[1], 1);
  const auto* gate32 = OpGetAlternative<Gate>(fgate3->gates[2]);
  ASSERT_NE(gate32, nullptr);
  EXPECT_EQ(gate32->kind, kGateHd);
  EXPECT_EQ(gate32->time, 10);
  EXPECT_EQ(gate32->qubits.size(), 1);
  EXPECT_EQ(gate32->qubits[0], 0);
  const auto* gate33 = OpGetAlternative<Gate>(fgate3->gates[3]);
  ASSERT_NE(gate33, nullptr);
  EXPECT_EQ(gate33->kind, kGateHd);
  EXPECT_EQ(gate33->time, 10);
  EXPECT_EQ(gate33->qubits.size(), 1);
  EXPECT_EQ(gate33->qubits[0], 1);

  const auto* fgate4 = OpGetAlternative<FusedGate>(fused_gates[4]);
  ASSERT_NE(fgate4, nullptr);
  EXPECT_EQ(fgate4->kind, kGateCZ);
  EXPECT_EQ(fgate4->time, 9);
  EXPECT_EQ(fgate4->qubits.size(), 2);
  EXPECT_EQ(fgate4->qubits[0], 2);
  EXPECT_EQ(fgate4->qubits[1], 3);
  EXPECT_EQ(fgate4->gates.size(), 4);
  const auto* gate40 = OpGetAlternative<Gate>(fgate4->gates[0]);
  ASSERT_NE(gate40, nullptr);
  EXPECT_EQ(gate40->kind, kGateX);
  EXPECT_EQ(gate40->time, 3);
  EXPECT_EQ(gate40->qubits.size(), 1);
  EXPECT_EQ(gate40->qubits[0], 3);
  const auto* gate41 = OpGetAlternative<Gate>(fgate4->gates[1]);
  ASSERT_NE(gate41, nullptr);
  EXPECT_EQ(gate41->kind, kGateCZ);
  EXPECT_EQ(gate41->time, 9);
  EXPECT_EQ(gate41->qubits.size(), 2);
  EXPECT_EQ(gate41->qubits[0], 2);
  EXPECT_EQ(gate41->qubits[1], 3);
  const auto* gate42 = OpGetAlternative<Gate>(fgate4->gates[2]);
  ASSERT_NE(gate42, nullptr);
  EXPECT_EQ(gate42->kind, kGateHd);
  EXPECT_EQ(gate42->time, 10);
  EXPECT_EQ(gate42->qubits.size(), 1);
  EXPECT_EQ(gate42->qubits[0], 2);
  const auto* gate43 = OpGetAlternative<Gate>(fgate4->gates[3]);
  ASSERT_NE(gate43, nullptr);
  EXPECT_EQ(gate43->kind, kGateHd);
  EXPECT_EQ(gate43->time, 10);
  EXPECT_EQ(gate43->qubits.size(), 1);
  EXPECT_EQ(gate43->qubits[0], 3);
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
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string2);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(2, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.ops.size(), 7);

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, circuit.ops);

  EXPECT_EQ(fused_gates.size(), 2);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 5);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);
  const auto* gate03 = OpGetAlternative<Gate>(fgate0->gates[3]);
  ASSERT_NE(gate03, nullptr);
  EXPECT_EQ(gate03->kind, kGateT);
  EXPECT_EQ(gate03->time, 2);
  EXPECT_EQ(gate03->qubits.size(), 1);
  EXPECT_EQ(gate03->qubits[0], 0);
  const auto* gate04 = OpGetAlternative<Gate>(fgate0->gates[4]);
  ASSERT_NE(gate04, nullptr);
  EXPECT_EQ(gate04->kind, kGateX);
  EXPECT_EQ(gate04->time, 2);
  EXPECT_EQ(gate04->qubits.size(), 1);
  EXPECT_EQ(gate04->qubits[0], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateHd);
  EXPECT_EQ(fgate1->time, 0);
  EXPECT_EQ(fgate1->qubits.size(), 1);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->gates.size(), 2);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateY);
  EXPECT_EQ(gate11->time, 2);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 2);
}

TEST(FuserBasicTest, OrphanedQubits2) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string2);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.ops.size(), 9);

  std::vector<unsigned> times_to_split_at{1, 4};

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(
      param, circuit.num_qubits, circuit.ops, times_to_split_at);

  EXPECT_EQ(fused_gates.size(), 4);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 3);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateHd);
  EXPECT_EQ(fgate1->time, 0);
  EXPECT_EQ(fgate1->qubits.size(), 1);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->gates.size(), 1);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateCZ);
  EXPECT_EQ(fgate2->time, 3);
  EXPECT_EQ(fgate2->qubits.size(), 2);
  EXPECT_EQ(fgate2->qubits[0], 1);
  EXPECT_EQ(fgate2->qubits[1], 2);
  EXPECT_EQ(fgate2->gates.size(), 3);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateX);
  EXPECT_EQ(gate20->time, 2);
  EXPECT_EQ(gate20->qubits.size(), 1);
  EXPECT_EQ(gate20->qubits[0], 1);
  const auto* gate21 = OpGetAlternative<Gate>(fgate2->gates[1]);
  ASSERT_NE(gate21, nullptr);
  EXPECT_EQ(gate21->kind, kGateY);
  EXPECT_EQ(gate21->time, 2);
  EXPECT_EQ(gate21->qubits.size(), 1);
  EXPECT_EQ(gate21->qubits[0], 2);
  const auto* gate22 = OpGetAlternative<Gate>(fgate2->gates[2]);
  ASSERT_NE(gate22, nullptr);
  EXPECT_EQ(gate22->kind, kGateCZ);
  EXPECT_EQ(gate22->time, 3);
  EXPECT_EQ(gate22->qubits.size(), 2);
  EXPECT_EQ(gate22->qubits[0], 1);
  EXPECT_EQ(gate22->qubits[1], 2);

  const auto* fgate3 = OpGetAlternative<FusedGate>(fused_gates[3]);
  ASSERT_NE(fgate3, nullptr);
  EXPECT_EQ(fgate3->kind, kGateT);
  EXPECT_EQ(fgate3->time, 2);
  EXPECT_EQ(fgate3->qubits.size(), 1);
  EXPECT_EQ(fgate3->qubits[0], 0);
  EXPECT_EQ(fgate3->gates.size(), 2);
  const auto* gate30 = OpGetAlternative<Gate>(fgate3->gates[0]);
  ASSERT_NE(gate30, nullptr);
  EXPECT_EQ(gate30->kind, kGateT);
  EXPECT_EQ(gate30->time, 2);
  EXPECT_EQ(gate30->qubits.size(), 1);
  EXPECT_EQ(gate30->qubits[0], 0);
  const auto* gate31 = OpGetAlternative<Gate>(fgate3->gates[1]);
  ASSERT_NE(gate31, nullptr);
  EXPECT_EQ(gate31->kind, kGateX);
  EXPECT_EQ(gate31->time, 4);
  EXPECT_EQ(gate31->qubits.size(), 1);
  EXPECT_EQ(gate31->qubits[0], 0);
}

TEST(FuserBasicTest, DecomposedQubitGate) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;
  using DecomposedGate = qsim::DecomposedGate<float>;

  using OperationD = detail::append_to_variant_t<Operation, DecomposedGate>;

  std::stringstream ss(circuit_string2);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(2, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 3);
  EXPECT_EQ(circuit.ops.size(), 7);

  Circuit<OperationD> circuitd;
  circuitd.num_qubits = circuit.num_qubits;
  circuitd.ops.resize(circuit.ops.size());

  circuitd.ops[0] = *OpGetAlternative<Gate>(circuit.ops[0]);
  circuitd.ops[1] = DecomposedGate{*OpGetAlternative<Gate>(circuit.ops[1])};
  circuitd.ops[2] = DecomposedGate{*OpGetAlternative<Gate>(circuit.ops[2])};
  circuitd.ops[3] = *OpGetAlternative<Gate>(circuit.ops[3]);
  circuitd.ops[4] = *OpGetAlternative<Gate>(circuit.ops[4]);
  circuitd.ops[5] = *OpGetAlternative<Gate>(circuit.ops[5]);
  circuitd.ops[6] = *OpGetAlternative<Gate>(circuit.ops[6]);

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  auto fused_gates = Fuser::FuseGates(param, circuitd.num_qubits, circuitd.ops);

  EXPECT_EQ(fused_gates.size(), 3);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateHd);
  EXPECT_EQ(fgate0->time, 0);
  EXPECT_EQ(fgate0->qubits.size(), 1);
  EXPECT_EQ(fgate0->qubits[0], 1);
  const auto* gate00 = OpGetAlternative<DecomposedGate>(fgate0->gates[0]);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateHd);
  EXPECT_EQ(fgate1->time, 0);
  EXPECT_EQ(fgate1->qubits.size(), 1);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->gates.size(), 2);
  const auto* gate10 = OpGetAlternative<DecomposedGate>(fgate1->gates[0]);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateY);
  EXPECT_EQ(gate11->time, 2);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 2);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateCZ);
  EXPECT_EQ(fgate2->time, 1);
  EXPECT_EQ(fgate2->qubits.size(), 2);
  EXPECT_EQ(fgate2->qubits[0], 0);
  EXPECT_EQ(fgate2->qubits[1], 1);
  EXPECT_EQ(fgate2->gates.size(), 4);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateHd);
  EXPECT_EQ(gate20->time, 0);
  EXPECT_EQ(gate20->qubits.size(), 1);
  EXPECT_EQ(gate20->qubits[0], 0);
  const auto* gate21 = OpGetAlternative<Gate>(fgate2->gates[1]);
  ASSERT_NE(gate21, nullptr);
  EXPECT_EQ(gate21->kind, kGateCZ);
  EXPECT_EQ(gate21->time, 1);
  EXPECT_EQ(gate21->qubits.size(), 2);
  EXPECT_EQ(gate21->qubits[0], 0);
  EXPECT_EQ(gate21->qubits[1], 1);
  const auto* gate22 = OpGetAlternative<Gate>(fgate2->gates[2]);
  ASSERT_NE(gate22, nullptr);
  EXPECT_EQ(gate22->kind, kGateT);
  EXPECT_EQ(gate22->time, 2);
  EXPECT_EQ(gate22->qubits.size(), 1);
  EXPECT_EQ(gate22->qubits[0], 0);
  const auto* gate23 = OpGetAlternative<Gate>(fgate2->gates[3]);
  ASSERT_NE(gate23, nullptr);
  EXPECT_EQ(gate23->kind, kGateX);
  EXPECT_EQ(gate23->time, 2);
  EXPECT_EQ(gate23->qubits.size(), 1);
  EXPECT_EQ(gate23->qubits[0], 1);
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
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string3);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 17);

  // Vector of pointers to gates.
  std::vector<const Operation*> pgates;
  pgates.reserve(circuit.ops.size());

  for (const auto& gate : circuit.ops) {
    pgates.push_back(&gate);
  }

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  const auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, pgates);

  EXPECT_EQ(fused_gates.size(), 11);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 3);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateHd);
  EXPECT_EQ(fgate1->time, 0);
  EXPECT_EQ(fgate1->qubits.size(), 1);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->gates.size(), 1);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateHd);
  EXPECT_EQ(fgate2->time, 0);
  EXPECT_EQ(fgate2->qubits.size(), 1);
  EXPECT_EQ(fgate2->qubits[0], 3);
  EXPECT_EQ(fgate2->gates.size(), 1);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateHd);
  EXPECT_EQ(gate20->time, 0);
  EXPECT_EQ(gate20->qubits.size(), 1);
  EXPECT_EQ(gate20->qubits[0], 3);

  const auto* fgate3 = OpGetAlternative<Measurement>(fused_gates[3]);
  EXPECT_EQ(fgate3->kind, kMeasurement);
  EXPECT_EQ(fgate3->time, 1);
  EXPECT_EQ(fgate3->qubits.size(), 2);
  EXPECT_EQ(fgate3->qubits[0], 2);
  EXPECT_EQ(fgate3->qubits[1], 3);

  const auto* fgate4 = OpGetAlternative<FusedGate>(fused_gates[4]);
  ASSERT_NE(fgate4, nullptr);
  EXPECT_EQ(fgate4->kind, kGateIS);
  EXPECT_EQ(fgate4->time, 2);
  EXPECT_EQ(fgate4->qubits.size(), 2);
  EXPECT_EQ(fgate4->qubits[0], 2);
  EXPECT_EQ(fgate4->qubits[1], 3);
  EXPECT_EQ(fgate4->gates.size(), 1);
  const auto* gate40 = OpGetAlternative<Gate>(fgate4->gates[0]);
  ASSERT_NE(gate40, nullptr);
  EXPECT_EQ(gate40->kind, kGateIS);
  EXPECT_EQ(gate40->time, 2);
  EXPECT_EQ(gate40->qubits.size(), 2);
  EXPECT_EQ(gate40->qubits[0], 2);
  EXPECT_EQ(gate40->qubits[1], 3);

  const auto* fgate5 = OpGetAlternative<FusedGate>(fused_gates[5]);
  ASSERT_NE(fgate5, nullptr);
  EXPECT_EQ(fgate5->kind, kGateCZ);
  EXPECT_EQ(fgate5->time, 3);
  EXPECT_EQ(fgate5->qubits.size(), 2);
  EXPECT_EQ(fgate5->qubits[0], 0);
  EXPECT_EQ(fgate5->qubits[1], 1);
  EXPECT_EQ(fgate5->gates.size(), 3);
  const auto* gate50 = OpGetAlternative<Gate>(fgate5->gates[0]);
  ASSERT_NE(gate50, nullptr);
  EXPECT_EQ(gate50->kind, kGateX);
  EXPECT_EQ(gate50->time, 2);
  EXPECT_EQ(gate50->qubits.size(), 1);
  EXPECT_EQ(gate50->qubits[0], 0);
  const auto* gate51 = OpGetAlternative<Gate>(fgate5->gates[1]);
  ASSERT_NE(gate51, nullptr);
  EXPECT_EQ(gate51->kind, kGateY);
  EXPECT_EQ(gate51->time, 2);
  EXPECT_EQ(gate51->qubits.size(), 1);
  EXPECT_EQ(gate51->qubits[0], 1);
  const auto* gate52 = OpGetAlternative<Gate>(fgate5->gates[2]);
  ASSERT_NE(gate52, nullptr);
  EXPECT_EQ(gate52->kind, kGateCZ);
  EXPECT_EQ(gate52->time, 3);
  EXPECT_EQ(gate52->qubits.size(), 2);
  EXPECT_EQ(gate52->qubits[0], 0);
  EXPECT_EQ(gate52->qubits[1], 1);

  const auto* fgate6 = OpGetAlternative<Measurement>(fused_gates[6]);
  EXPECT_EQ(fgate6->kind, kMeasurement);
  EXPECT_EQ(fgate6->time, 3);
  EXPECT_EQ(fgate6->qubits.size(), 2);
  EXPECT_EQ(fgate6->qubits[0], 2);
  EXPECT_EQ(fgate6->qubits[1], 3);

  const auto* fgate7 = OpGetAlternative<FusedGate>(fused_gates[7]);
  ASSERT_NE(fgate7, nullptr);
  EXPECT_EQ(fgate7->kind, kGateIS);
  EXPECT_EQ(fgate7->time, 4);
  EXPECT_EQ(fgate7->qubits.size(), 2);
  EXPECT_EQ(fgate7->qubits[0], 2);
  EXPECT_EQ(fgate7->qubits[1], 3);
  EXPECT_EQ(fgate7->gates.size(), 1);
  const auto* gate70 = OpGetAlternative<Gate>(fgate7->gates[0]);
  ASSERT_NE(gate70, nullptr);
  EXPECT_EQ(gate70->kind, kGateIS);
  EXPECT_EQ(gate70->time, 4);
  EXPECT_EQ(gate70->qubits.size(), 2);
  EXPECT_EQ(gate70->qubits[0], 2);
  EXPECT_EQ(gate70->qubits[1], 3);

  const auto* fgate8 = OpGetAlternative<FusedGate>(fused_gates[8]);
  ASSERT_NE(fgate8, nullptr);
  EXPECT_EQ(fgate8->kind, kGateX);
  EXPECT_EQ(fgate8->time, 4);
  EXPECT_EQ(fgate8->qubits.size(), 1);
  EXPECT_EQ(fgate8->qubits[0], 0);
  EXPECT_EQ(fgate8->gates.size(), 1);
  const auto* gate80 = OpGetAlternative<Gate>(fgate8->gates[0]);
  ASSERT_NE(gate80, nullptr);
  EXPECT_EQ(gate80->kind, kGateX);
  EXPECT_EQ(gate80->time, 4);
  EXPECT_EQ(gate80->qubits.size(), 1);
  EXPECT_EQ(gate80->qubits[0], 0);

  const auto* fgate9 = OpGetAlternative<FusedGate>(fused_gates[9]);
  ASSERT_NE(fgate9, nullptr);
  EXPECT_EQ(fgate9->kind, kGateY);
  EXPECT_EQ(fgate9->time, 4);
  EXPECT_EQ(fgate9->qubits.size(), 1);
  EXPECT_EQ(fgate9->qubits[0], 1);
  EXPECT_EQ(fgate9->gates.size(), 1);
  const auto* gate90 = OpGetAlternative<Gate>(fgate9->gates[0]);
  ASSERT_NE(gate90, nullptr);
  EXPECT_EQ(gate90->kind, kGateY);
  EXPECT_EQ(gate90->time, 4);
  EXPECT_EQ(gate90->qubits.size(), 1);
  EXPECT_EQ(gate90->qubits[0], 1);

  const auto* fgate10 = OpGetAlternative<Measurement>(fused_gates[10]);
  EXPECT_EQ(fgate10->kind, kMeasurement);
  EXPECT_EQ(fgate10->time, 5);
  EXPECT_EQ(fgate10->qubits.size(), 4);
  EXPECT_EQ(fgate10->qubits[0], 2);
  EXPECT_EQ(fgate10->qubits[1], 3);
  EXPECT_EQ(fgate10->qubits[2], 0);
  EXPECT_EQ(fgate10->qubits[3], 1);
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
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using ControlledGate = qsim::ControlledGate<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::stringstream ss(circuit_string4);
  Circuit<Operation> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 5);
  EXPECT_EQ(circuit.ops.size(), 13);

  // Vector of pointers to gates.
  std::vector<const Operation*> pgates;
  pgates.reserve(circuit.ops.size());

  for (const auto& gate : circuit.ops) {
    pgates.push_back(&gate);
  }

  using Fuser = BasicGateFuser<IO>;
  Fuser::Parameter param;
  const auto fused_gates = Fuser::FuseGates(param, circuit.num_qubits, pgates);

  EXPECT_EQ(fused_gates.size(), 8);

  const auto* fgate0 = OpGetAlternative<FusedGate>(fused_gates[0]);
  ASSERT_NE(fgate0, nullptr);
  EXPECT_EQ(fgate0->kind, kGateCZ);
  EXPECT_EQ(fgate0->time, 1);
  EXPECT_EQ(fgate0->qubits.size(), 2);
  EXPECT_EQ(fgate0->qubits[0], 0);
  EXPECT_EQ(fgate0->qubits[1], 1);
  EXPECT_EQ(fgate0->gates.size(), 3);
  const auto* gate00 = OpGetAlternative<Gate>(fgate0->gates[0]);
  ASSERT_NE(gate00, nullptr);
  EXPECT_EQ(gate00->kind, kGateHd);
  EXPECT_EQ(gate00->time, 0);
  EXPECT_EQ(gate00->qubits.size(), 1);
  EXPECT_EQ(gate00->qubits[0], 0);
  const auto* gate01 = OpGetAlternative<Gate>(fgate0->gates[1]);
  ASSERT_NE(gate01, nullptr);
  EXPECT_EQ(gate01->kind, kGateHd);
  EXPECT_EQ(gate01->time, 0);
  EXPECT_EQ(gate01->qubits.size(), 1);
  EXPECT_EQ(gate01->qubits[0], 1);
  const auto* gate02 = OpGetAlternative<Gate>(fgate0->gates[2]);
  ASSERT_NE(gate02, nullptr);
  EXPECT_EQ(gate02->kind, kGateCZ);
  EXPECT_EQ(gate02->time, 1);
  EXPECT_EQ(gate02->qubits.size(), 2);
  EXPECT_EQ(gate02->qubits[0], 0);
  EXPECT_EQ(gate02->qubits[1], 1);

  const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
  ASSERT_NE(fgate1, nullptr);
  EXPECT_EQ(fgate1->kind, kGateCZ);
  EXPECT_EQ(fgate1->time, 1);
  EXPECT_EQ(fgate1->qubits.size(), 2);
  EXPECT_EQ(fgate1->qubits[0], 2);
  EXPECT_EQ(fgate1->qubits[1], 3);
  EXPECT_EQ(fgate1->gates.size(), 4);
  const auto* gate10 = OpGetAlternative<Gate>(fgate1->gates[0]);
  ASSERT_NE(gate10, nullptr);
  EXPECT_EQ(gate10->kind, kGateHd);
  EXPECT_EQ(gate10->time, 0);
  EXPECT_EQ(gate10->qubits.size(), 1);
  EXPECT_EQ(gate10->qubits[0], 2);
  const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
  ASSERT_NE(gate11, nullptr);
  EXPECT_EQ(gate11->kind, kGateHd);
  EXPECT_EQ(gate11->time, 0);
  EXPECT_EQ(gate11->qubits.size(), 1);
  EXPECT_EQ(gate11->qubits[0], 3);
  const auto* gate12 = OpGetAlternative<Gate>(fgate1->gates[2]);
  ASSERT_NE(gate12, nullptr);
  EXPECT_EQ(gate12->kind, kGateCZ);
  EXPECT_EQ(gate12->time, 1);
  EXPECT_EQ(gate12->qubits.size(), 2);
  EXPECT_EQ(gate12->qubits[0], 2);
  EXPECT_EQ(gate12->qubits[1], 3);
  const auto* gate13 = OpGetAlternative<Gate>(fgate1->gates[3]);
  ASSERT_NE(gate13, nullptr);
  EXPECT_EQ(gate13->kind, kGateHd);
  EXPECT_EQ(gate13->time, 3);
  EXPECT_EQ(gate13->qubits.size(), 1);
  EXPECT_EQ(gate13->qubits[0], 3);

  const auto* fgate2 = OpGetAlternative<FusedGate>(fused_gates[2]);
  ASSERT_NE(fgate2, nullptr);
  EXPECT_EQ(fgate2->kind, kGateHd);
  EXPECT_EQ(fgate2->time, 0);
  EXPECT_EQ(fgate2->qubits.size(), 1);
  EXPECT_EQ(fgate2->qubits[0], 4);
  EXPECT_EQ(fgate2->gates.size(), 1);
  const auto* gate20 = OpGetAlternative<Gate>(fgate2->gates[0]);
  ASSERT_NE(gate20, nullptr);
  EXPECT_EQ(gate20->kind, kGateHd);
  EXPECT_EQ(gate20->time, 0);
  EXPECT_EQ(gate20->qubits.size(), 1);
  EXPECT_EQ(gate20->qubits[0], 4);

  const auto* fgate3 = OpGetAlternative<ControlledGate>(fused_gates[3]);
  EXPECT_EQ(fgate3->kind, kGateHd);
  EXPECT_EQ(fgate3->time, 2);
  EXPECT_EQ(fgate3->qubits.size(), 1);
  EXPECT_EQ(fgate3->qubits[0], 2);
  EXPECT_EQ(fgate3->controlled_by.size(), 3);
  EXPECT_EQ(fgate3->controlled_by[0], 0);
  EXPECT_EQ(fgate3->controlled_by[1], 1);
  EXPECT_EQ(fgate3->controlled_by[2], 4);
  EXPECT_EQ(fgate3->cmask, 7);

  const auto* fgate4 = OpGetAlternative<FusedGate>(fused_gates[4]);
  ASSERT_NE(fgate4, nullptr);
  EXPECT_EQ(fgate4->kind, kGateHd);
  EXPECT_EQ(fgate4->time, 3);
  EXPECT_EQ(fgate4->qubits.size(), 1);
  EXPECT_EQ(fgate4->qubits[0], 0);
  EXPECT_EQ(fgate4->gates.size(), 1);
  const auto* gate40 = OpGetAlternative<Gate>(fgate4->gates[0]);
  ASSERT_NE(gate40, nullptr);
  EXPECT_EQ(gate40->kind, kGateHd);
  EXPECT_EQ(gate40->time, 3);
  EXPECT_EQ(gate40->qubits.size(), 1);
  EXPECT_EQ(gate40->qubits[0], 0);

  const auto* fgate5 = OpGetAlternative<FusedGate>(fused_gates[5]);
  ASSERT_NE(fgate5, nullptr);
  EXPECT_EQ(fgate5->kind, kGateHd);
  EXPECT_EQ(fgate5->time, 3);
  EXPECT_EQ(fgate5->qubits.size(), 1);
  EXPECT_EQ(fgate5->qubits[0], 1);
  EXPECT_EQ(fgate5->gates.size(), 1);
  const auto* gate50 = OpGetAlternative<Gate>(fgate5->gates[0]);
  ASSERT_NE(gate50, nullptr);
  EXPECT_EQ(gate50->kind, kGateHd);
  EXPECT_EQ(gate50->time, 3);
  EXPECT_EQ(gate50->qubits.size(), 1);
  EXPECT_EQ(gate50->qubits[0], 1);

  const auto* fgate6 = OpGetAlternative<FusedGate>(fused_gates[6]);
  ASSERT_NE(fgate6, nullptr);
  EXPECT_EQ(fgate6->kind, kGateHd);
  EXPECT_EQ(fgate6->time, 3);
  EXPECT_EQ(fgate6->qubits.size(), 1);
  EXPECT_EQ(fgate6->qubits[0], 2);
  EXPECT_EQ(fgate6->gates.size(), 1);
  const auto* gate60 = OpGetAlternative<Gate>(fgate6->gates[0]);
  ASSERT_NE(gate60, nullptr);
  EXPECT_EQ(gate60->kind, kGateHd);
  EXPECT_EQ(gate60->time, 3);
  EXPECT_EQ(gate60->qubits.size(), 1);
  EXPECT_EQ(gate60->qubits[0], 2);

  const auto* fgate7 = OpGetAlternative<FusedGate>(fused_gates[7]);
  ASSERT_NE(fgate7, nullptr);
  EXPECT_EQ(fgate7->kind, kGateHd);
  EXPECT_EQ(fgate7->time, 3);
  EXPECT_EQ(fgate7->qubits.size(), 1);
  EXPECT_EQ(fgate7->qubits[0], 4);
  EXPECT_EQ(fgate7->gates.size(), 1);
  const auto* gate70 = OpGetAlternative<Gate>(fgate7->gates[0]);
  ASSERT_NE(gate70, nullptr);
  EXPECT_EQ(gate70->kind, kGateHd);
  EXPECT_EQ(gate70->time, 3);
  EXPECT_EQ(gate70->qubits.size(), 1);
  EXPECT_EQ(gate70->qubits[0], 4);
}

TEST(FuserBasicTest, SmallCircuits) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = BasicGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateGPh<float>::Create(0, -1, 0),
    };

    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateGPh<float>::Create(0, -1, 0),
      GateGPh<float>::Create(0, -1, 0),
    };

    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateGPh<float>::Create(0, -1, 0),
    };

    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Operation> circuit = {
      GateZ<float>::Create(0, 0).ControlledBy({1}, {1}),
      GateCZ<float>::Create(1, 0, 1),
      GateGPh<float>::Create(1, -1, 0),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 2);

    const auto* fgate1 = OpGetAlternative<FusedGate<float>>(fused_gates[1]);
    const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
    ASSERT_NE(gate11, nullptr);
    EXPECT_EQ(gate11, OpGetAlternative<Gate>(circuit[2]));
  }
}

TEST(FuserBasicTest, ValidTimeOrder) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = BasicGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 8;
    auto gate1 = GateZ<float>::Create(1, 2);
    auto gate2 = GateZ<float>::Create(2, 5);

    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      MakeControlledGate(gate1, {1}),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(2, 0, 1),
      GateCZ<float>::Create(1, 3, 4),
      GateCZ<float>::Create(2, 2, 3),
      GateCZ<float>::Create(3, 1, 2),
      MakeControlledGate(gate2, {4}),
      GateCZ<float>::Create(3, 3, 4),
      GateCZ<float>::Create(5, 0, 1),
      GateCZ<float>::Create(4, 2, 3),
      GateCZ<float>::Create(5, 4, 5),
      GateCZ<float>::Create(4, 6, 7),
    };

    const auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 14);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(1, 1, 2),
      GateCZ<float>::Create(0, 4, 5),
      GateCZ<float>::Create(1, 3, 4),
      CreateMeasurement(2, {0, 1, 2}),
      CreateMeasurement(2, {4, 5}),
      GateCZ<float>::Create(3, 0, 1),
      GateCZ<float>::Create(3, 2, 3),
      GateCZ<float>::Create(4, 1, 2),
      GateCZ<float>::Create(3, 4, 5),
      GateCZ<float>::Create(4, 3, 4),
    };

    const auto fused_gates = Fuser::FuseGates<Operation>(
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

    const auto fused_gates = Fuser::FuseGates<Gate>(
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

    const auto fused_gates = Fuser::FuseGates<Gate>(
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
    const auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), time_boundary);

    EXPECT_EQ(fused_gates.size(), 6);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 1, 2),
      GateCZ<float>::Create(1, 0, 3),
      CreateMeasurement(3, {1, 2}),
      GateCZ<float>::Create(3, 0, 3),
      GateCZ<float>::Create(5, 1, 2),
      GateCZ<float>::Create(4, 0, 3),
    };

    const auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 7);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(1, 0, 1),
      GateGPh<float>::Create(2, -1, 0),
    };

    std::vector<unsigned> time_boundary = {1};
    const auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 2);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(1, 0, 1),
      GateGPh<float>::Create(0, -1, 0),
    };

    const auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }
}

TEST(FuserBasicTest, InvalidTimeOrder) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = BasicGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates<Gate>(
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

    auto fused_gates = Fuser::FuseGates<Gate>(
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
    auto fused_gates = Fuser::FuseGates<Gate>(
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
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), time_boundary);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      CreateMeasurement(2, {0, 3}),
      GateCZ<float>::Create(1, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      CreateMeasurement(1, {1, 2}),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    auto gate = GateZ<float>::Create(1, 1);

    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      GateCZ<float>::Create(2, 0, 3),
      MakeControlledGate(gate, {3}),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    auto gate = GateZ<float>::Create(2, 1);

    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      GateCZ<float>::Create(0, 2, 3),
      MakeControlledGate(gate, {3}),
      GateCZ<float>::Create(1, 0, 3),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(1, 0, 1),
      GateGPh<float>::Create(0, -1, 0),
    };

    std::vector<unsigned> time_boundary = {1};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

TEST(FuserBasicTest, QubitsOutOfRange) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = BasicGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      GateCZ<float>::Create(0, 0, 3),
      GateCZ<float>::Create(0, 1, 2),
    };

    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 3;
    auto gate = GateZ<float>::Create(0, 2);

    std::vector<Operation> circuit = {
      GateCZ<float>::Create(0, 0, 1),
      MakeControlledGate(gate, {3}),
    };

    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
