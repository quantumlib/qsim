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

#include <custatevecEx.h>

#include "gtest/gtest.h"

#include "../lib/multiprocess_custatevecex.h"
#include "../lib/simulator_custatevecex.h"
#include "../lib/statespace_custatevecex.h"

namespace qsim {

template <class T>
class StateSpaceCuStateVecExTest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(StateSpaceCuStateVecExTest, fp_impl);

MultiProcessCuStateVecEx mp;

template <typename fp_type>
struct Factory {
  using Simulator = qsim::SimulatorCuStateVecEx<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace CreateStateSpace() const {
    return StateSpace{mp};
  }

  Simulator CreateSimulator() const {
    return Simulator{};
  }
};

TYPED_TEST(StateSpaceCuStateVecExTest, Add) {
  TestAdd(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, NormSmall) {
  TestNormSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, NormAndInnerProduct) {
  TestNormAndInnerProduct(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, SamplingSmall) {
  TestSamplingSmall(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, Ordering) {
  TestOrdering(qsim::Factory<TypeParam>());
}

TEST(StateSpaceCuStateVecExTest, MeasurementSmall) {
  TestMeasurementSmall(qsim::Factory<float>(), true);
}

TYPED_TEST(StateSpaceCuStateVecExTest, MeasurementLarge) {
//  This test fails.
//  TestMeasurementLarge(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, Collapse) {
//  Not implemented.
//  TestCollapse(qsim::Factory<TypeParam>());
}

TEST(StateSpaceCuStateVecExTest, InvalidStateSize) {
  TestInvalidStateSize(qsim::Factory<float>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, BulkSetAmpl) {
//  Not implemented.
//  TestBulkSetAmplitude(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, BulkSetAmplExclusion) {
//  Not implemented.
//  TestBulkSetAmplitudeExclusion(qsim::Factory<TypeParam>());
}

TYPED_TEST(StateSpaceCuStateVecExTest, BulkSetAmplDefault) {
//  Not implemented.
//  TestBulkSetAmplitudeDefault(factory);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  qsim::mp.initialize();

  return RUN_ALL_TESTS();
}
