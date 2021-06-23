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

#include "../lib/mps_simulator.h"

#include "../lib/formux.h"
#include "gtest/gtest.h"

namespace qsim {

namespace mps {

namespace {

TEST(MPSSimulator, Create) { auto sim = MPSSimulator<For, float>(1); }

TEST(MPSSimulator, Apply1RightArbitrary) {
  // Apply an arbitrary matrix to the last qubit triggers [bond_dim, 2].
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto state = ss.CreateMPS(10, 4);
  ss.SetMPSZero(state);
  auto offset = ss.GetBlockOffset(state, 9);
  // Completely fill final block.
  for (unsigned i = offset; i < ss.Size(state); i++) {
    state.get()[i] = float(i - offset);
  }

  std::vector<float> mat = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  sim.ApplyGate({9}, mat.data(), state);

  // Ensure other blocks unchanged.
  for (unsigned i = 0; i < offset; i++) {
    auto expected = 0.;
    if (i == 0 || (i - 16) % 64 == 0) {
      expected = 1.0;
    }
    ASSERT_NEAR(state.get()[i], expected, 1e-5);
  }
  // [[-0.8 +1.8j -1.6 +4.2j]
  //  [-1.6 +5.8j -2.4+14.6j]
  //  [-2.4 +9.8j -3.2+25.j ]
  //  [-3.2+13.8j -4. +35.4j]]
  ASSERT_NEAR(state.get()[offset + 0], -0.8, 1e-5);
  ASSERT_NEAR(state.get()[offset + 1], 1.8, 1e-5);
  ASSERT_NEAR(state.get()[offset + 2], -1.6, 1e-5);
  ASSERT_NEAR(state.get()[offset + 3], 4.2, 1e-5);
  ASSERT_NEAR(state.get()[offset + 4], -1.6, 1e-5);
  ASSERT_NEAR(state.get()[offset + 5], 5.8, 1e-5);
  ASSERT_NEAR(state.get()[offset + 6], -2.4, 1e-5);
  ASSERT_NEAR(state.get()[offset + 7], 14.6, 1e-5);

  ASSERT_NEAR(state.get()[offset + 8], -2.4, 1e-5);
  ASSERT_NEAR(state.get()[offset + 9], 9.8, 1e-5);
  ASSERT_NEAR(state.get()[offset + 10], -3.2, 1e-5);
  ASSERT_NEAR(state.get()[offset + 11], 25., 1e-5);
  ASSERT_NEAR(state.get()[offset + 12], -3.2, 1e-5);
  ASSERT_NEAR(state.get()[offset + 13], 13.8, 1e-5);
  ASSERT_NEAR(state.get()[offset + 14], -4., 1e-5);
  ASSERT_NEAR(state.get()[offset + 15], 35.4, 1e-5);
}

TEST(MPSSimulator, Apply1LeftArbitrary) {
  // Apply a matrix to the first qubit triggers [2, bond_dim].
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto state = ss.CreateMPS(10, 4);
  ss.SetMPSZero(state);
  std::vector<float> matrix = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  // Completely fill first block.
  for (unsigned i = 0; i < ss.GetBlockOffset(state, 1); i++) {
    state.get()[i] = float(i);
  }
  sim.ApplyGate({0}, matrix.data(), state);

  auto offset = ss.GetBlockOffset(state, 1);

  // Ensure other blocks unchanged.
  for (unsigned i = offset; i < ss.Size(state); i++) {
    auto expected = 0.;
    if ((i - 16) % 64 == 0) {
      expected = 1.0;
    }
    ASSERT_NEAR(state.get()[i], expected, 1e-5);
  }

  //  [[-1.4 +6.j  -1.8 +8.j  -2.2+10.j  -2.6+12.j ]
  //   [-2.2+13.2j -2.6+18.4j -3. +23.6j -3.4+28.8j]]
  ASSERT_NEAR(state.get()[0], -1.4, 1e-5);
  ASSERT_NEAR(state.get()[1], 6, 1e-5);
  ASSERT_NEAR(state.get()[2], -1.8, 1e-5);
  ASSERT_NEAR(state.get()[3], 8, 1e-5);
  ASSERT_NEAR(state.get()[4], -2.2, 1e-5);
  ASSERT_NEAR(state.get()[5], 10, 1e-5);
  ASSERT_NEAR(state.get()[6], -2.6, 1e-5);
  ASSERT_NEAR(state.get()[7], 12, 1e-5);

  ASSERT_NEAR(state.get()[8], -2.2, 1e-5);
  ASSERT_NEAR(state.get()[9], 13.2, 1e-5);
  ASSERT_NEAR(state.get()[10], -2.6, 1e-5);
  ASSERT_NEAR(state.get()[11], 18.4, 1e-5);
  ASSERT_NEAR(state.get()[12], -3.0, 1e-5);
  ASSERT_NEAR(state.get()[13], 23.6, 1e-5);
  ASSERT_NEAR(state.get()[14], -3.4, 1e-5);
  ASSERT_NEAR(state.get()[15], 28.8, 1e-5);
}

TEST(MPSSimulator, Apply1InteriorArbitrary) {
  // Apply a matrix to the second qubit. Triggers [bond_dim, 2, bond_dim].
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);
  auto state = ss.CreateMPS(10, 4);
  ss.SetMPSZero(state);
  std::vector<float> matrix = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  // Completely fill second block.
  auto l_offset = ss.GetBlockOffset(state, 1);
  auto r_offset = ss.GetBlockOffset(state, 2);
  for (unsigned i = l_offset; i < r_offset; i++) {
    state.get()[i] = float(i - l_offset);
  }
  sim.ApplyGate({1}, matrix.data(), state);

  // Ensure other blocks unchanged.
  for (unsigned i = 0; i < l_offset; i++) {
    auto expected = 0.;
    if (i == 0 || (i - 16) % 64 == 0) {
      expected = 1.0;
    }
    ASSERT_NEAR(state.get()[i], expected, 1e-5);
  }
  for (unsigned i = r_offset; i < ss.Size(state); i++) {
    auto expected = 0.;
    if ((i - 16) % 64 == 0) {
      expected = 1.0;
    }
    ASSERT_NEAR(state.get()[i], expected, 1e-5);
  }

  // Look at [0, ... , ...]
  //  [[-1.4 +6.j  -1.8 +8.j  -2.2+10.j  -2.6+12.j ]
  //   [-2.2+13.2j -2.6+18.4j -3. +23.6j -3.4+28.8j]]
  ASSERT_NEAR(state.get()[l_offset + 0], -1.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 1], 6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 2], -1.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 3], 8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 4], -2.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 5], 10, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 6], -2.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 7], 12, 1e-5);

  ASSERT_NEAR(state.get()[l_offset + 8], -2.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 9], 13.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 10], -2.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 11], 18.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 12], -3.0, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 13], 23.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 14], -3.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 15], 28.8, 1e-5);

  // Look at [1, ... , ...]
  //  [[-4.6+22.j  -5. +24.j  -5.4+26.j  -5.8+28.j ]
  //   [-5.4+54.8j -5.8+60.j  -6.2+65.2j -6.6+70.4j]]
  ASSERT_NEAR(state.get()[l_offset + 16], -4.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 17], 22, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 18], -5, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 19], 24, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 20], -5.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 21], 26, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 22], -5.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 23], 28, 1e-5);

  ASSERT_NEAR(state.get()[l_offset + 24], -5.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 25], 54.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 26], -5.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 27], 60, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 28], -6.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 29], 65.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 30], -6.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 31], 70.4, 1e-5);

  // Look at [2, ... , ...]
  //  [[-7.8 +38.j  -8.2 +40.j  -8.6 +42.j  -9.  +44.j ]
  //   [-8.6 +96.4j -9. +101.6j -9.4+106.8j -9.8+112.j ]]
  ASSERT_NEAR(state.get()[l_offset + 32], -7.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 33], 38, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 34], -8.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 35], 40, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 36], -8.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 37], 42, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 38], -9.0, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 39], 44, 1e-5);

  ASSERT_NEAR(state.get()[l_offset + 40], -8.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 41], 96.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 42], -9.0, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 43], 101.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 44], -9.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 45], 106.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 46], -9.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 47], 112, 1e-5);

  // Look at [3, ... , ...]
  //  [[-11.  +54.j  -11.4 +56.j  -11.8 +58.j  -12.2 +60.j ]
  //   [-11.8+138.j  -12.2+143.2j -12.6+148.4j -13. +153.6j]]
  ASSERT_NEAR(state.get()[l_offset + 48], -11, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 49], 54, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 50], -11.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 51], 56, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 52], -11.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 53], 58, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 54], -12.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 55], 60, 1e-5);

  ASSERT_NEAR(state.get()[l_offset + 56], -11.8, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 57], 138, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 58], -12.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 59], 143.2, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 60], -12.6, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 61], 148.4, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 62], -13, 1e-5);
  ASSERT_NEAR(state.get()[l_offset + 63], 153.6, 1e-5);
}

}  // namespace
}  // namespace mps
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
