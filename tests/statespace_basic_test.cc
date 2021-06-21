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

#include "gtest/gtest.h"

#ifdef _OPENMP
#include "../lib/parfor.h"
#endif
#include "../lib/seqfor.h"
#include "../lib/simulator_basic.h"
#include "../lib/statespace_basic.h"

namespace qsim {

template <class T>
class StateSpaceBasicTest : public testing::Test {};

using ::testing::Types;
#ifdef _OPENMP
typedef Types<ParallelFor, SequentialFor> for_impl;
#else
typedef Types<SequentialFor> for_impl;
#endif

template <typename For>
struct Factory {
  using Simulator = SimulatorBasic<For, float>;
  using StateSpace = typename Simulator::StateSpace;

  static StateSpace CreateStateSpace() {
    return StateSpace(2);
  }

  static Simulator CreateSimulator() {
    return Simulator(2);
  }
};

TYPED_TEST_SUITE(StateSpaceBasicTest, for_impl);

TYPED_TEST(StateSpaceBasicTest, Add) {
  TestAdd(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, NormSmall) {
  TestNormSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, NormAndInnerProduct) {
  TestNormAndInnerProduct(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, SamplingSmall) {
  TestSamplingSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, Ordering) {
  TestOrdering(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, MeasurementSmall) {
  TestMeasurementSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, MeasurementLarge) {
  TestMeasurementLarge(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, Collapse) {
  TestCollapse(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, InvalidStateSize) {
  TestInvalidStateSize(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, BulkSetAmpl) {
  TestBulkSetAmplitude(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, BulkSetAmplExclude) {
  TestBulkSetAmplitudeExclusion(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, BulkSetAmplDefault) {
  TestBulkSetAmplitudeDefault(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceBasicTest, ThreadThrashing) {
  TestThreadThrashing<StateSpaceBasic<TypeParam, float>>();
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
