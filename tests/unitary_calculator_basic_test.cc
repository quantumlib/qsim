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
#include "../lib/unitary_calculator_basic.h"

namespace qsim {
namespace unitary {
namespace {

TEST(UnitaryCalculatorBasicTest, ApplyGate1) {
  TestApplyGate1<UnitaryCalculatorBasic<For, float>>();
}

TEST(UnitaryCalculatorBasicTest, ApplyControlledGate1) {
  TestApplyControlledGate1<UnitaryCalculatorBasic<For, float>>();
}

TEST(UnitaryCalculatorBasicTest, ApplyGate2) {
  TestApplyGate2<UnitaryCalculatorBasic<For, float>>();
}

TEST(UnitaryCalculatorBasicTest, ApplyControlledGate2) {
  TestApplyControlledGate2<UnitaryCalculatorBasic<For, float>>();
}

TEST(UnitaryCalculatorBasicTest, ApplyFusedGate) {
  TestApplyFusedGate<UnitaryCalculatorBasic<For, float>>();
}

TEST(UnitaryCalculatorBasicTest, ApplyGates) {
  TestApplyGates<UnitaryCalculatorBasic<For, float>>(false);
  TestApplyGates<UnitaryCalculatorBasic<For, double>>(true);
}

TEST(UnitaryCalculatorBasicTest, ApplyControlledGates) {
  TestApplyControlledGates<UnitaryCalculatorBasic<For, float>>(false);
  TestApplyControlledGates<UnitaryCalculatorBasic<For, double>>(true);
}

TEST(UnitaryCalculatorBasicTest, SmallCircuits) {
  TestSmallCircuits<UnitaryCalculatorBasic<For, float>>();
}

}  // namespace
}  // namespace unitary
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
