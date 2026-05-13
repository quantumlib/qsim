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
#include "../lib/simulator_neon.h"
#include "../lib/statespace_neon.h"

namespace qsim {

template <class T>
class StateSpaceNEONTest : public testing::Test {};

using ::testing::Types;
#ifdef _OPENMP
typedef Types<ParallelFor, SequentialFor> for_impl;
#else
typedef Types<SequentialFor> for_impl;
#endif

template <typename For>
struct Factory {
  using Simulator = SimulatorNEON<For>;
  using StateSpace = typename Simulator::StateSpace;

  static StateSpace CreateStateSpace() {
    return StateSpace(2);
  }

  static Simulator CreateSimulator() {
    return Simulator(2);
  }
};

TYPED_TEST_SUITE(StateSpaceNEONTest, for_impl);

TYPED_TEST(StateSpaceNEONTest, Add) {
  TestAdd(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, NormSmall) {
  TestNormSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, NormAndInnerProduct) {
  TestNormAndInnerProduct(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, SamplingSmall) {
  TestSamplingSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, Ordering) {
  TestOrdering(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, MeasurementSmall) {
  TestMeasurementSmall(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, MeasurementLarge) {
  TestMeasurementLarge(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, Collapse) {
  TestCollapse(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, InvalidStateSize) {
  TestInvalidStateSize(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, BulkSetAmpl) {
  TestBulkSetAmplitude(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, BulkSetAmplExclude) {
  TestBulkSetAmplitudeExclusion(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, BulkSetAmplDefault) {
  TestBulkSetAmplitudeDefault(Factory<TypeParam>());
}

TYPED_TEST(StateSpaceNEONTest, ThreadThrashing) {
  TestThreadThrashing<StateSpaceNEON<TypeParam>>();
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
