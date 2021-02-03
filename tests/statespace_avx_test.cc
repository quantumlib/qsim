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
#include "../lib/simulator_avx.h"
#include "../lib/statespace_avx.h"

namespace qsim {

template <class T>
class StateSpaceAVXTest : public testing::Test {};

using ::testing::Types;
#ifdef _OPENMP
typedef Types<ParallelFor, SequentialFor> for_impl;
#else
typedef Types<SequentialFor> for_impl;
#endif

TYPED_TEST_SUITE(StateSpaceAVXTest, for_impl);

TYPED_TEST(StateSpaceAVXTest, Add) {
  TestAdd<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, NormSmall) {
  TestNormSmall<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, NormAndInnerProduct) {
  TestNormAndInnerProduct<SimulatorAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, SamplingSmall) {
  TestSamplingSmall<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference<SimulatorAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, Ordering) {
  TestOrdering<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, MeasurementSmall) {
  TestMeasurementSmall<StateSpaceAVX<TypeParam>, TypeParam>();
}

TYPED_TEST(StateSpaceAVXTest, MeasurementLarge) {
  TestMeasurementLarge<SimulatorAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, InvalidStateSize) {
  TestInvalidStateSize<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, BulkSetAmpl) {
  TestBulkSetAmplitude<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, BulkSetAmplExclude) {
  TestBulkSetAmplitudeExclusion<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, BulkSetAmplDefault) {
  TestBulkSetAmplitudeDefault<StateSpaceAVX<TypeParam>>();
}

TYPED_TEST(StateSpaceAVXTest, ThreadThrashing) {
  TestThreadThrashing<StateSpaceAVX<TypeParam>>();
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
