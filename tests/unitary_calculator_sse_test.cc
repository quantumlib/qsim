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
#include "../lib/unitary_calculator_sse.h"

namespace qsim {
namespace unitary {
namespace {

TEST(UnitaryCalculatorSSETest, ApplyGate1) {
  TestApplyGate1<UnitaryCalculatorSSE<For>>();
}

TEST(UnitaryCalculatorSSETest, ApplyControlledGate1) {
  TestApplyControlledGate1<UnitaryCalculatorSSE<For>>();
}

TEST(UnitaryCalculatorSSETest, ApplyGate2) {
  TestApplyGate2<UnitaryCalculatorSSE<For>>();
}

TEST(UnitaryCalculatorSSETest, ApplyControlledGate2) {
  TestApplyControlledGate2<UnitaryCalculatorSSE<For>>();
}

TEST(UnitaryCalculatorSSETest, ApplyFusedGate) {
  TestApplyFusedGate<UnitaryCalculatorSSE<For>>();
}

TEST(UnitaryCalculatorSSETest, ApplyGates) {
  TestApplyGates<UnitaryCalculatorSSE<For>>(false);
}

TEST(UnitaryCalculatorSSETest, ApplyControlledGates) {
  TestApplyControlledGates<UnitaryCalculatorSSE<For>>(false);
}

TEST(UnitaryCalculatorSSETest, SmallCircuits) {
  TestSmallCircuits<UnitaryCalculatorSSE<For>>();
}

}  // namespace
}  // namespace unitary
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
