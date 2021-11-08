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

#include <cublas_v2.h>
#include <custatevec.h>

#include <type_traits>

#include "gtest/gtest.h"

#include "../lib/simulator_custatevec.h"

namespace qsim {

template <class T>
class SimulatorCuStateVecTest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(SimulatorCuStateVecTest, fp_impl);

template <typename fp_type>
struct Factory {
  using Simulator = qsim::SimulatorCuStateVec<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  Factory() {
    ErrorCheck(cublasCreate(&cublas_handle));
    ErrorCheck(custatevecCreate(&custatevec_handle));
  }

  ~Factory() {
    ErrorCheck(cublasDestroy(cublas_handle));
    ErrorCheck(custatevecDestroy(custatevec_handle));
  }

  StateSpace CreateStateSpace() const {
    return StateSpace(cublas_handle, custatevec_handle);
  }

  Simulator CreateSimulator() const {
    return Simulator(custatevec_handle);
  }

  cublasHandle_t cublas_handle;
  custatevecHandle_t custatevec_handle;
};

TYPED_TEST(SimulatorCuStateVecTest, ApplyGate1) {
  TestApplyGate1(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, ApplyGate2) {
  TestApplyGate2(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, ApplyGate3) {
  TestApplyGate3(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, ApplyGate5) {
  TestApplyGate5(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, CircuitWithControlledGates) {
  TestCircuitWithControlledGates(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, CircuitWithControlledGatesDagger) {
  TestCircuitWithControlledGatesDagger(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, MultiQubitGates) {
  TestMultiQubitGates(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, ControlledGates) {
  bool high_precision = std::is_same<TypeParam, double>::value;
  TestControlledGates(qsim::Factory<TypeParam>(), high_precision);
}

TYPED_TEST(SimulatorCuStateVecTest, ExpectationValue1) {
  TestExpectationValue1(qsim::Factory<TypeParam>());
}

TYPED_TEST(SimulatorCuStateVecTest, ExpectationValue2) {
  TestExpectationValue2(qsim::Factory<TypeParam>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
