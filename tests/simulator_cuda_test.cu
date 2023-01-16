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

#include <type_traits>

#include "gtest/gtest.h"

#include "../lib/simulator_cuda.h"

namespace qsim {

template <class T>
class SimulatorCUDATest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(SimulatorCUDATest, fp_impl);

template <typename fp_type>
struct Factory {
  using Simulator = qsim::SimulatorCUDA<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  Factory(const typename StateSpace::Parameter& param) : param(param) {}

  StateSpace CreateStateSpace() const {
    return StateSpace(param);
  }

  Simulator CreateSimulator() const {
    return Simulator();
  }

  typename StateSpace::Parameter param;
};

TYPED_TEST(SimulatorCUDATest, ApplyGate1) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestApplyGate1(factory);
}

TYPED_TEST(SimulatorCUDATest, ApplyGate2) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestApplyGate2(factory);
}

TYPED_TEST(SimulatorCUDATest, ApplyGate3) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestApplyGate3(factory);
}

TYPED_TEST(SimulatorCUDATest, ApplyGate5) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestApplyGate5(factory);
}

TYPED_TEST(SimulatorCUDATest, CircuitWithControlledGates) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestCircuitWithControlledGates(factory);
}

TYPED_TEST(SimulatorCUDATest, CircuitWithControlledGatesDagger) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestCircuitWithControlledGatesDagger(factory);
}

TYPED_TEST(SimulatorCUDATest, MultiQubitGates) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestMultiQubitGates(factory);
}

TYPED_TEST(SimulatorCUDATest, ControlledGates) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  bool high_precision = std::is_same<TypeParam, double>::value;
  TestControlledGates(factory, high_precision);
}

TYPED_TEST(SimulatorCUDATest, GlobalPhaseGate) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestGlobalPhaseGate(factory);
}

TYPED_TEST(SimulatorCUDATest, ExpectationValue1) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestExpectationValue1(factory);
}

TYPED_TEST(SimulatorCUDATest, ExpectationValue2) {
  using Factory = qsim::Factory<TypeParam>;
  typename Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestExpectationValue2(factory);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
