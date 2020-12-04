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

#include "simulator_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/simulator_sse.h"

namespace qsim {

TEST(SimulatorSSETest, ApplyGate1) {
  TestApplyGate1<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ApplyGate2) {
  TestApplyGate2<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ApplyGate3) {
  TestApplyGate3<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ApplyGate5) {
  TestApplyGate5<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ApplyControlGate) {
  TestApplyControlGate<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, MultiQubitGates) {
  TestMultiQubitGates<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ExpectationValue1) {
  TestExpectationValue1<SimulatorSSE<For>>();
}

TEST(SimulatorSSETest, ExpectationValue2) {
  TestExpectationValue2<SimulatorSSE<For>>();
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
