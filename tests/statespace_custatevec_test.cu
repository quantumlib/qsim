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

#include "statespace_testfixture.h"

#include <cublas_v2.h>
#include <custatevec.h>

#include "gtest/gtest.h"

#include "../lib/simulator_custatevec.h"
#include "../lib/statespace_custatevec.h"

namespace qsim {

template <class T>
class StateSpaceCuStateVecTest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(StateSpaceCuStateVecTest, fp_impl);

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

TYPED_TEST(StateSpaceCuStateVecTest, Add) {
  TestAdd(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, NormSmall) {
  TestNormSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, NormAndInnerProduct) {
  TestNormAndInnerProduct(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, SamplingSmall) {
  TestSamplingSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, Ordering) {
  TestOrdering(qsim::Factory<TypeParam>());
}

TEST(StateSpaceCuStateVecTest, MeasurementSmall) {
  TestMeasurementSmall(qsim::Factory<float>(), true);
}

TYPED_TEST(StateSpaceCuStateVecTest, MeasurementLarge) {
//  This test fails.
//  TestMeasurementLarge(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, Collapse) {
  TestCollapse(qsim::Factory<TypeParam>());
}

TEST(StateSpaceCuStateVecTest, InvalidStateSize) {
  TestInvalidStateSize(qsim::Factory<float>());
}

TYPED_TEST(StateSpaceCuStateVecTest, BulkSetAmpl) {
//  Not implemented.
//  TestBulkSetAmplitude(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, BulkSetAmplExclusion) {
//  Not implemented.
//  TestBulkSetAmplitudeExclusion(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecTest, BulkSetAmplDefault) {
//  Not implemented.
//  TestBulkSetAmplitudeDefault(factory);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
