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

#ifdef _OPENMP
#include "../lib/parfor.h"
#endif
#include "../lib/seqfor.h"
#include "../lib/simulator_sse.h"

namespace qsim {

template <class T>
class SimulatorSSETest : public testing::Test {};

using ::testing::Types;
#ifdef _OPENMP
typedef Types<ParallelFor, SequentialFor> for_impl;
#else
typedef Types<SequentialFor> for_impl;
#endif

template <typename For>
struct Factory {
  using Simulator = SimulatorSSE<For>;
  using StateSpace = typename Simulator::StateSpace;

  static StateSpace CreateStateSpace() {
    return StateSpace(2);
  }

  static Simulator CreateSimulator() {
    return Simulator(2);
  }
};

TYPED_TEST_SUITE(SimulatorSSETest, for_impl);

TYPED_TEST(SimulatorSSETest, ApplyGate1) {
  TestApplyGate1(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, ApplyGate2) {
  TestApplyGate2(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, ApplyGate3) {
  TestApplyGate3(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, ApplyGate5) {
  TestApplyGate5(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, CircuitWithControlledGates) {
  TestCircuitWithControlledGates(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, CircuitWithControlledGatesDagger) {
  TestCircuitWithControlledGatesDagger(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, MultiQubitGates) {
  TestMultiQubitGates(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, ControlledGates) {
  TestControlledGates(Factory<TypeParam>(), false);
}

TYPED_TEST(SimulatorSSETest, ExpectationValue1) {
  TestExpectationValue1(Factory<TypeParam>());
}

TYPED_TEST(SimulatorSSETest, ExpectationValue2) {
  TestExpectationValue2(Factory<TypeParam>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
