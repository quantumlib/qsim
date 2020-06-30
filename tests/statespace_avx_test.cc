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

#include "../lib/formux.h"
#include "../lib/simulator_avx.h"
#include "../lib/statespace_avx.h"

namespace qsim {

TEST(StateSpaceAVXTest, NormSmall) {
  TestNormSmall<StateSpaceAVX<For>>();
}

TEST(StateSpaceAVXTest, NormAndInnerProductSmall) {
  TestNormAndInnerProductSmall<StateSpaceAVX<For>>();
}

TEST(StateSpaceAVXTest, NormAndInnerProduct) {
  TestNormAndInnerProduct<SimulatorAVX<For>>();
}

TEST(StateSpaceAVXTest, SamplingSmall) {
  TestSamplingSmall<StateSpaceAVX<For>>();
}

TEST(StateSpaceAVXTest, SamplingCrossEntropyDifference) {
  TestSamplingCrossEntropyDifference<SimulatorAVX<For>>();
}

TEST(StateSpaceAVXTest, Ordering) {
  TestOrdering<StateSpaceAVX<For>>();
}

TEST(StateSpaceAVXTest, MeasurementSmall) {
  TestMeasurementSmall<StateSpaceAVX<For>, For>();
}

TEST(StateSpaceAVXTest, MeasurementLarge) {
  TestMeasurementLarge<SimulatorAVX<For>>();
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
