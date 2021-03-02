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

#include "unitary_calculator_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/unitary_calculator_avx.h"

namespace qsim {
namespace unitary {
namespace {

TEST(UnitaryCalculatorAVXTest, ApplyGate1) {
  TestApplyGate1<UnitaryCalculatorAVX<For>>();
}

TEST(UnitaryCalculatorAVXTest, ApplyControlledGate1) {
  TestApplyControlledGate1<UnitaryCalculatorAVX<For>>();
}

TEST(UnitaryCalculatorAVXTest, ApplyGate2) {
  TestApplyGate2<UnitaryCalculatorAVX<For>>();
}

TEST(UnitaryCalculatorAVXTest, ApplyControlledGate2) {
  TestApplyControlledGate2<UnitaryCalculatorAVX<For>>();
}

TEST(UnitaryCalculatorAVXTest, ApplyFusedGate) {
  TestApplyFusedGate<UnitaryCalculatorAVX<For>>();
}

TEST(UnitaryCalculatorAVXTest, ApplyGates) {
  TestApplyGates<UnitaryCalculatorAVX<For>>(false);
}

TEST(UnitaryCalculatorAVXTest, ApplyControlledGates) {
  TestApplyControlledGates<UnitaryCalculatorAVX<For>>(false);
}

TEST(UnitaryCalculatorAVXTest, SmallCircuits) {
  TestSmallCircuits<UnitaryCalculatorAVX<For>>();
}

}  // namespace
}  // namespace unitary
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
