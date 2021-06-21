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

#include "../lib/mps_statespace.h"

#include "../lib/formux.h"
#include "gtest/gtest.h"

namespace qsim {

namespace mps {

namespace {

TEST(MPSStateSpaceTest, Create) {
  auto ss = MPSStateSpace<For, float>(1);
  auto mps = ss.CreateMPS(5, 8);
  EXPECT_EQ(mps.num_qubits(), 5);
  EXPECT_EQ(mps.bond_dim(), 8);
}

TEST(MPSStateSpaceTest, BlockOffset) {
  auto ss = MPSStateSpace<For, float>(1);
  auto mps = ss.CreateMPS(5, 8);
  for (int i = 0; i < ss.Size(mps); i++) {
    mps.get()[i] = i;
  }

  ASSERT_EQ(ss.GetBlockOffset(mps, 0), 0);
  ASSERT_EQ(ss.GetBlockOffset(mps, 1), 32);
  ASSERT_EQ(ss.GetBlockOffset(mps, 2), 256 + 32);
  ASSERT_EQ(ss.GetBlockOffset(mps, 3), 512 + 32);
  ASSERT_EQ(ss.GetBlockOffset(mps, 4), 768 + 32);
}

TEST(MPSStateSpaceTest, SetZero) {
  auto ss = MPSStateSpace<For, float>(1);
  auto mps = ss.CreateMPS(4, 8);
  for (int i = 0; i < ss.Size(mps); i++) {
    mps.get()[i] = i;
  }
  ss.SetMPSZero(mps);
  for (int i = 0; i < ss.Size(mps); i++) {
    auto expected = 0.0;
    if (i == 0 || i == 32 || i == 256 + 32 || i == 512 + 32) {
      expected = 1;
    }
    EXPECT_NEAR(mps.get()[i], expected, 1e-5);
  }
}

TEST(MPSStateSpaceTest, Copy) {
  auto ss = MPSStateSpace<For, float>(1);
  auto mps = ss.CreateMPS(10, 8);
  auto mps2 = ss.CreateMPS(10, 8);
  auto mps3 = ss.CreateMPS(10, 4);
  for (int i = 0; i < ss.Size(mps); i++) {
    mps.get()[i] = i;
  }
  ASSERT_FALSE(ss.CopyMPS(mps, mps3));
  ss.CopyMPS(mps, mps2);
  for (int i = 0; i < ss.Size(mps); i++) {
    EXPECT_NEAR(mps.get()[i], mps2.get()[i], 1e-5);
  }
}

}  // namespace
}  // namespace mps
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
