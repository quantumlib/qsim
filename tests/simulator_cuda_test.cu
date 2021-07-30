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

  Factory(const typename StateSpace::Parameter& param1,
          const typename Simulator::Parameter& param2)
      : param1(param1), param2(param2) {}

  StateSpace CreateStateSpace() const {
    return StateSpace(param1);
  }

  Simulator CreateSimulator() const {
    return Simulator(param2);
  }

  typename StateSpace::Parameter param1;
  typename Simulator::Parameter param2;
};

TYPED_TEST(SimulatorCUDATest, ApplyGate1) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestApplyGate1(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, ApplyGate2) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestApplyGate2(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, ApplyGate3) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestApplyGate3(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, ApplyGate5) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestApplyGate5(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, CircuitWithControlledGates) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestCircuitWithControlledGates(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, CircuitWithControlledGatesDagger) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestCircuitWithControlledGatesDagger(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, MultiQubitGates) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestMultiQubitGates(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, ControlledGates) {
  using Factory = qsim::Factory<TypeParam>;

  bool high_precision = std::is_same<TypeParam, double>::value;

  for (unsigned num_threads : {64, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestControlledGates(factory, high_precision);
  }
}

TYPED_TEST(SimulatorCUDATest, ExpectationValue1) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {32, 64, 128, 256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestExpectationValue1(factory);
  }
}

TYPED_TEST(SimulatorCUDATest, ExpectationValue2) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_threads : {256}) {
    typename Factory::Simulator::Parameter param;
    param.num_threads = num_threads;

    Factory factory(typename Factory::StateSpace::Parameter(), param);

    TestExpectationValue2(factory);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
