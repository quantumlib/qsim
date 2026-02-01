// Copyright 2025 Google LLC. All Rights Reserved.
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

#include <custatevecEx.h>

#include <type_traits>

#include "gtest/gtest.h"

#include "../lib/multiprocess_custatevecex.h"
#include "../lib/simulator_custatevecex.h"

namespace qsim {

template <class T>
class SimulatorCuStateVecExTest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(SimulatorCuStateVecExTest, fp_impl);

MultiProcessCuStateVecEx mp;

template <typename fp_type>
struct Factory {
  using Simulator = qsim::SimulatorCuStateVecEx<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace CreateStateSpace() const {
    typename StateSpace::Parameter param;
    param.num_devices = 2;
    return StateSpace{mp, param};
  }

  Simulator CreateSimulator() const {
    return Simulator{};
  }
};

TYPED_TEST(SimulatorCuStateVecExTest, ApplyGate1) {
  TestApplyGate1(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ApplyGate2) {
  TestApplyGate2(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ApplyGate3) {
  TestApplyGate3(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ApplyGate5) {
  TestApplyGate5(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, CircuitWithControlledGates) {
  TestCircuitWithControlledGates(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, CircuitWithControlledGatesDagger) {
  TestCircuitWithControlledGatesDagger(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, MultiQubitGates) {
  TestMultiQubitGates(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ControlledGates) {
  bool high_precision = std::is_same<TypeParam, double>::value;
  TestControlledGates(qsim::Factory<TypeParam>(), high_precision, true);
}

TYPED_TEST(SimulatorCuStateVecExTest, GlobalPhaseGate) {
  TestGlobalPhaseGate(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ExpectationValue1) {
  TestExpectationValue1(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecExTest, ExpectationValue2) {
  TestExpectationValue2(qsim::Factory<TypeParam>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  qsim::mp.initialize(qsim::MultiProcessCuStateVecEx::Parameter{});

  return RUN_ALL_TESTS();
}
