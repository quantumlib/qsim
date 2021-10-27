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
#include "../lib/gate_appl.h"
#include "../lib/gates_cirq.h"
#include "../lib/gates_qsim.h"
#include "gtest/gtest.h"

namespace qsim {

namespace mps {

namespace {

TEST(MPSSimulator, Create) {
  MPSSimulator<For, float>(1);
}

TEST(MPSSimulator, Apply1RightArbitrary) {
  // Apply an arbitrary matrix to the last qubit triggers [bond_dim, 2].
  //   |     |     |
  //   |     |   +-+-+
  //   |     |   | U |
  //   |     |   +-+-+
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto state = ss.Create(10, 4);
  ss.SetStateZero(state);
  auto offset = ss.GetBlockOffset(state, 9);
  // Completely fill final block.
  for (unsigned i = offset; i < ss.Size(state); ++i) {
    state.get()[i] = float(i - offset);
  }

  std::vector<float> mat = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  sim.ApplyGate({9}, mat.data(), state);

  // Ensure other blocks unchanged.
  for (unsigned i = 0; i < offset; ++i) {
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
  //   |     |     |
  // +-+-+   |     |
  // | U |   |     |
  // +-+-+   |     |
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto state = ss.Create(10, 4);
  ss.SetStateZero(state);
  std::vector<float> matrix = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  // Completely fill first block.
  for (unsigned i = 0; i < ss.GetBlockOffset(state, 1); ++i) {
    state.get()[i] = float(i);
  }
  sim.ApplyGate({0}, matrix.data(), state);

  auto offset = ss.GetBlockOffset(state, 1);

  // Ensure other blocks unchanged.
  for (unsigned i = offset; i < ss.Size(state); ++i) {
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
  //   |     |     |
  //   |   +-+-+   |
  //   |   | U |   |
  //   |   +-+-+   |
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);
  auto state = ss.Create(10, 4);
  ss.SetStateZero(state);
  std::vector<float> matrix = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  // Completely fill second block.
  auto l_offset = ss.GetBlockOffset(state, 1);
  auto r_offset = ss.GetBlockOffset(state, 2);
  for (unsigned i = l_offset; i < r_offset; ++i) {
    state.get()[i] = float(i - l_offset);
  }
  sim.ApplyGate({1}, matrix.data(), state);

  // Ensure other blocks unchanged.
  for (unsigned i = 0; i < l_offset; ++i) {
    auto expected = 0.;
    if (i == 0 || (i - 16) % 64 == 0) {
      expected = 1.0;
    }
    ASSERT_NEAR(state.get()[i], expected, 1e-5);
  }
  for (unsigned i = r_offset; i < ss.Size(state); ++i) {
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

TEST(MPSSimulator, Apply2Left01) {
  // Compute the state vector of:
  //   |     |     |
  // +-+-----+-+   |
  // |Syc  Gate|   |
  // +-+-----+-+   |
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  // Where 0, 1, 2 are in an initial highly entagled (random) state.
  // Tests left boundary contraction behavior.
  // Note: Since SVD is not unique, comparisons are more easily made against
  // the statevector and not the mps state itself.
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto mps = ss.Create(3, 4);
  memset(mps.get(), 0, ss.RawSize(mps));
  mps.get()[0] = -0.35590581027809776;
  mps.get()[1] = 0.026818375951314005;
  mps.get()[2] = 0.8727655590119396;
  mps.get()[3] = 0.3330047088601117;
  mps.get()[8] = -0.6775160618958234;
  mps.get()[9] = -0.6431047824209531;
  mps.get()[10] = -0.1296437424681659;
  mps.get()[11] = -0.33253671532297485;
  mps.get()[16] = 0.5941347479820251;
  mps.get()[17] = 0.19108928740024567;
  mps.get()[18] = 0.585273027420044;
  mps.get()[19] = -0.24035339057445526;
  mps.get()[24] = 0.38375747203826904;
  mps.get()[25] = 0.15926049649715424;
  mps.get()[26] = -0.2938160300254822;
  mps.get()[27] = -0.12502236664295197;
  mps.get()[32] = -0.46030130982398987;
  mps.get()[33] = -0.40587466955184937;
  mps.get()[34] = 0.41750138998031616;
  mps.get()[35] = -0.3110700249671936;
  mps.get()[40] = 0.2459115833044052;
  mps.get()[41] = -0.027767818421125412;
  mps.get()[42] = -0.4402020573616028;
  mps.get()[43] = -0.181321382522583;
  mps.get()[80] = -0.40654852986335754;
  mps.get()[81] = 0.0;
  mps.get()[82] = 0.7775961756706238;
  mps.get()[83] = 0.0723680704832077;
  mps.get()[84] = 0.42058154940605164;
  mps.get()[85] = 0.0;
  mps.get()[86] = 0.21800334751605988;
  mps.get()[87] = 0.02028878591954708;

  std::vector<float> mat = {
      1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,  0.0,
      0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,  0.0,  0.0,
      0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.866, -0.5,
  };
  sim.ApplyGate({0, 1}, mat.data(), mps);

  float wf[32];
  ss.ToWaveFunction(mps, wf);
  EXPECT_NEAR(wf[0], 0.30834898352622986, 1e-4);
  EXPECT_NEAR(wf[1], 0.21436986327171326, 1e-4);
  EXPECT_NEAR(wf[2], -0.2765581011772156, 1e-4);
  EXPECT_NEAR(wf[3], -0.4715091586112976, 1e-4);
  EXPECT_NEAR(wf[4], -0.0068932194262743, 1e-4);
  EXPECT_NEAR(wf[5], 0.1537550389766693, 1e-4);
  EXPECT_NEAR(wf[6], -0.34593719244003296, 1e-4);
  EXPECT_NEAR(wf[7], 0.40202534198760986, 1e-4);
  EXPECT_NEAR(wf[8], -0.11738976091146469, 1e-4);
  EXPECT_NEAR(wf[9], 0.12454932928085327, 1e-4);
  EXPECT_NEAR(wf[10], -0.048086442053318024, 1e-4);
  EXPECT_NEAR(wf[11], -0.022116247564554214, 1e-4);
  EXPECT_NEAR(wf[12], 0.29326894879341125, 1e-4);
  EXPECT_NEAR(wf[13], 0.24929752945899963, 1e-4);
  EXPECT_NEAR(wf[14], -0.21864625811576843, 1e-4);
  EXPECT_NEAR(wf[15], -0.16468186676502228, 1e-4);
}

TEST(MPSSimulator, Apply2Right12) {
  // Compute the state vector of:
  //   |     |     |
  //   |   +-+-----+-+
  //   |   |Syc  Gate|
  //   |   +-+-----+-+
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  // Where 0, 1, 2 are in an initial highly entagled (random) state.
  // Tests right boundary contraction behavior.
  // Note: Since SVD is not unique, comparisons are more easily made against
  // the statevector and not the mps state itself.
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto mps = ss.Create(3, 4);
  memset(mps.get(), 0, ss.RawSize(mps));
  mps.get()[0] = -0.35590581027809776;
  mps.get()[1] = 0.026818375951314005;
  mps.get()[2] = 0.8727655590119396;
  mps.get()[3] = 0.3330047088601117;
  mps.get()[8] = -0.6775160618958234;
  mps.get()[9] = -0.6431047824209531;
  mps.get()[10] = -0.1296437424681659;
  mps.get()[11] = -0.33253671532297485;
  mps.get()[16] = 0.5941347479820251;
  mps.get()[17] = 0.19108928740024567;
  mps.get()[18] = 0.585273027420044;
  mps.get()[19] = -0.24035339057445526;
  mps.get()[24] = 0.38375747203826904;
  mps.get()[25] = 0.15926049649715424;
  mps.get()[26] = -0.2938160300254822;
  mps.get()[27] = -0.12502236664295197;
  mps.get()[32] = -0.46030130982398987;
  mps.get()[33] = -0.40587466955184937;
  mps.get()[34] = 0.41750138998031616;
  mps.get()[35] = -0.3110700249671936;
  mps.get()[40] = 0.2459115833044052;
  mps.get()[41] = -0.027767818421125412;
  mps.get()[42] = -0.4402020573616028;
  mps.get()[43] = -0.181321382522583;
  mps.get()[80] = -0.40654852986335754;
  mps.get()[81] = 0.0;
  mps.get()[82] = 0.7775961756706238;
  mps.get()[83] = 0.0723680704832077;
  mps.get()[84] = 0.42058154940605164;
  mps.get()[85] = 0.0;
  mps.get()[86] = 0.21800334751605988;
  mps.get()[87] = 0.02028878591954708;

  std::vector<float> mat = {
      1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,  0.0,
      0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,  0.0,  0.0,
      0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.866, -0.5,
  };
  sim.ApplyGate({1, 2}, mat.data(), mps);

  float wf[32];
  ss.ToWaveFunction(mps, wf);
  EXPECT_NEAR(wf[0], 0.30834902875645503, 1e-4);
  EXPECT_NEAR(wf[1], 0.2143698948599676, 1e-4);
  EXPECT_NEAR(wf[2], -0.11738977370009586, 1e-4);
  EXPECT_NEAR(wf[3], 0.12454935773362025, 1e-4);
  EXPECT_NEAR(wf[4], -0.47150921436100324, 1e-4);
  EXPECT_NEAR(wf[5], 0.2765580906536361, 1e-4);
  EXPECT_NEAR(wf[6], -0.00488996377601076, 1e-4);
  EXPECT_NEAR(wf[7], -0.052702222730185794, 1e-4);
  EXPECT_NEAR(wf[8], -0.15375504635790666, 1e-4);
  EXPECT_NEAR(wf[9], -0.006893208509005258, 1e-4);
  EXPECT_NEAR(wf[10], 0.3625325032891465, 1e-4);
  EXPECT_NEAR(wf[11], -0.12932957404571366, 1e-4);
  EXPECT_NEAR(wf[12], -0.3459372670125525, 1e-4);
  EXPECT_NEAR(wf[13], 0.4020253863039828, 1e-4);
  EXPECT_NEAR(wf[14], -0.21864624250067458, 1e-4);
  EXPECT_NEAR(wf[15], -0.16468189647310463, 1e-4);
}

TEST(MPSSimulator, Apply2Middle) {
  // Compute the state vector of:
  //   |     |     |     |
  //   |   +-+-----+-+   |
  //   |   |Syc  Gate|   |
  //   |   +-+-----+-+   |
  //   |     |     |     |
  // +-+-+ +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 +-+ 3 |
  // +---+ +---+ +---+ +---+
  // Where 0, 1, 2, 3 are in an initial highly entagled (random) state.
  // Tests boundaryless contraction behavior.
  // Note: Since SVD is not unique, comparisons are more easily made against
  // the statevector and not the mps state itself.
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto mps = ss.Create(4, 4);
  memset(mps.get(), 0, ss.RawSize(mps));
  mps.get()[0] = 0.7542437229597659;
  mps.get()[1] = 0.6320228540989201;
  mps.get()[2] = 0.1053532754040888;
  mps.get()[3] = 0.1434023655259462;
  mps.get()[8] = -0.1590242421535213;
  mps.get()[9] = -0.07984248439908945;
  mps.get()[10] = 0.7494392669999452;
  mps.get()[11] = 0.6377125027619668;
  mps.get()[16] = -0.22326465839130194;
  mps.get()[17] = -0.14802095790152967;
  mps.get()[18] = -0.7649273889911162;
  mps.get()[19] = 0.02342604699359434;
  mps.get()[20] = -0.27469206222952186;
  mps.get()[21] = -0.08516083433827198;
  mps.get()[22] = 0.3124756536081633;
  mps.get()[23] = 0.40277893114445384;
  mps.get()[24] = 0.1668126952115;
  mps.get()[25] = -0.6539200272717096;
  mps.get()[26] = 0.14852789666588434;
  mps.get()[27] = -0.16053397437320288;
  mps.get()[28] = -0.281740319243122;
  mps.get()[29] = -0.1774515411995222;
  mps.get()[30] = 0.4461465756950806;
  mps.get()[31] = -0.43222332846278144;
  mps.get()[32] = -0.13784245504574277;
  mps.get()[33] = 0.19659276591001018;
  mps.get()[34] = 0.4133421283756894;
  mps.get()[35] = 0.045212225231994585;
  mps.get()[36] = -0.8070948761311186;
  mps.get()[37] = -0.18600837901768108;
  mps.get()[38] = -0.10433763588711567;
  mps.get()[39] = 0.2693831530995169;
  mps.get()[40] = 0.5646522513457826;
  mps.get()[41] = -0.31036009410606136;
  mps.get()[42] = -0.23726315224787997;
  mps.get()[43] = 0.37056167241400584;
  mps.get()[44] = -0.25778185246992447;
  mps.get()[45] = 0.23231983702835832;
  mps.get()[46] = -0.5203807548766195;
  mps.get()[47] = 0.0038171159375407493;
  mps.get()[80] = -0.5495447176154851;
  mps.get()[81] = 0.0;
  mps.get()[82] = 0.22345295911822455;
  mps.get()[83] = -0.2728085507671099;
  mps.get()[84] = 0.0;
  mps.get()[85] = -9.330321032080985e-19;
  mps.get()[88] = -0.31797840847894143;
  mps.get()[89] = -0.43700313396907214;
  mps.get()[90] = -0.6308944060466369;
  mps.get()[91] = 0.20172528631281222;
  mps.get()[96] = -0.28022414449539923;
  mps.get()[97] = 0.0;
  mps.get()[98] = -0.04564618493522886;
  mps.get()[99] = -0.13058507325541344;
  mps.get()[100] = 0.0;
  mps.get()[101] = 5.36965589512873e-18;
  mps.get()[104] = 0.12194748032111768;
  mps.get()[105] = -0.4329782261603227;
  mps.get()[106] = 0.5430100970957932;
  mps.get()[107] = 0.06308187242715302;
  mps.get()[112] = -0.27928789505064155;
  mps.get()[113] = 0.0;
  mps.get()[114] = 0.1459631558719182;
  mps.get()[115] = 0.2775242271446053;
  mps.get()[116] = 5.1762131694603757e-17;
  mps.get()[117] = 3.2351332309127348e-18;
  mps.get()[120] = -0.05849653911507401;
  mps.get()[121] = 0.17380053379711788;
  mps.get()[122] = 0.08592497660828376;
  mps.get()[123] = 0.005820907919166205;
  mps.get()[128] = 0.08143433455290194;
  mps.get()[129] = 0.0;
  mps.get()[130] = 0.08903710395704752;
  mps.get()[131] = 0.05424906551273037;
  mps.get()[132] = 1.8116593011540607e-17;
  mps.get()[133] = 0.0;
  mps.get()[136] = -0.0552944001865287;
  mps.get()[137] = -0.06109363349983873;
  mps.get()[138] = 0.021214685571085405;
  mps.get()[139] = 0.04342293877681552;
  mps.get()[144] = -0.08290449155693787;
  mps.get()[145] = 0.3438100212145986;
  mps.get()[146] = 0.707739859156722;
  mps.get()[147] = -0.25136020408744075;
  mps.get()[148] = 0.47673412610059657;
  mps.get()[149] = 0.16477707955556076;
  mps.get()[150] = 0.057634996208717065;
  mps.get()[151] = 0.2304230685138138;

  std::vector<float> mat = {
      1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,  0.0,
      0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,  0.0,  0.0,
      0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.866, -0.5,
  };
  sim.ApplyGate({1, 2}, mat.data(), mps);

  float wf[128];
  ss.ToWaveFunction(mps, wf);
  EXPECT_NEAR(wf[0], -0.1982501457335573, 1e-4);
  EXPECT_NEAR(wf[1], 0.04522418081945127, 1e-4);
  EXPECT_NEAR(wf[2], 0.2896809705635635, 1e-4);
  EXPECT_NEAR(wf[3], 0.19277793971803137, 1e-4);
  EXPECT_NEAR(wf[4], -0.2614901084249643, 1e-4);
  EXPECT_NEAR(wf[5], -0.06528771452621676, 1e-4);
  EXPECT_NEAR(wf[6], 0.28773299777716393, 1e-4);
  EXPECT_NEAR(wf[7], 0.05651018009083913, 1e-4);
  EXPECT_NEAR(wf[8], -0.19262807240218052, 1e-4);
  EXPECT_NEAR(wf[9], 0.1480003324653067, 1e-4);
  EXPECT_NEAR(wf[10], 0.19906593043170975, 1e-4);
  EXPECT_NEAR(wf[11], 0.04953515397048365, 1e-4);
  EXPECT_NEAR(wf[12], -0.03561662367651478, 1e-4);
  EXPECT_NEAR(wf[13], 0.05464182463287305, 1e-4);
  EXPECT_NEAR(wf[14], -0.4090652266110749, 1e-4);
  EXPECT_NEAR(wf[15], 0.060959884879063723, 1e-4);
  EXPECT_NEAR(wf[16], 0.06215780255106787, 1e-4);
  EXPECT_NEAR(wf[17], -0.0546146507913291, 1e-4);
  EXPECT_NEAR(wf[18], 0.13328897145501709, 1e-4);
  EXPECT_NEAR(wf[19], -0.0023857963255580015, 1e-4);
  EXPECT_NEAR(wf[20], -0.08992925221102638, 1e-4);
  EXPECT_NEAR(wf[21], -0.17003990865425744, 1e-4);
  EXPECT_NEAR(wf[22], -0.09053910810077165, 1e-4);
  EXPECT_NEAR(wf[23], 0.09081144408861042, 1e-4);
  EXPECT_NEAR(wf[24], 0.23364296468006465, 1e-4);
  EXPECT_NEAR(wf[25], -0.11470612682283252, 1e-4);
  EXPECT_NEAR(wf[26], -0.12561881844347136, 1e-4);
  EXPECT_NEAR(wf[27], -0.30675870597826327, 1e-4);
  EXPECT_NEAR(wf[28], -0.2993801455524698, 1e-4);
  EXPECT_NEAR(wf[29], 0.03211453341738496, 1e-4);
  EXPECT_NEAR(wf[30], -0.19529420117137705, 1e-4);
  EXPECT_NEAR(wf[31], -0.13440315911542725, 1e-4);
}

TEST(MPSSimulator, OneTwoQubitFuzz) {
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);
  using Gate = qsim::Cirq::GateCirq<float>;
  using YPowGate = qsim::Cirq::YPowGate<float>;
  using FSimGate = qsim::Cirq::FSimGate<float>;

  unsigned bond_dim = 4;
  unsigned num_qubits = 6;
  std::vector<Gate> gates;
  gates.reserve(32);

  gates.push_back(YPowGate::Create(0, 0, 0.1));
  gates.push_back(YPowGate::Create(0, 1, 0.2));
  gates.push_back(YPowGate::Create(0, 2, 0.3));
  gates.push_back(YPowGate::Create(0, 3, 0.4));
  gates.push_back(YPowGate::Create(0, 4, 0.5));
  gates.push_back(YPowGate::Create(0, 5, 0.6));

  gates.push_back(FSimGate::Create(1, 0, 1, 1.1, 0.8));
  gates.push_back(FSimGate::Create(1, 2, 3, 0.2, 0.9));
  gates.push_back(FSimGate::Create(1, 4, 5, 0.3, 0.4));

  gates.push_back(YPowGate::Create(2, 0, 0.88));
  gates.push_back(YPowGate::Create(2, 1, 1.8));
  gates.push_back(YPowGate::Create(2, 2, -0.3));
  gates.push_back(YPowGate::Create(2, 3, 0.6));
  gates.push_back(YPowGate::Create(2, 4, -1.2));
  gates.push_back(YPowGate::Create(2, 5, 2.2));

  gates.push_back(FSimGate::Create(3, 1, 2, 0.5, -0.9));
  gates.push_back(FSimGate::Create(3, 3, 4, -4.1, 0.345));

  auto mps = ss.Create(num_qubits, bond_dim);
  ss.SetStateZero(mps);
  for (const auto &gate : gates) {
    ApplyGate(sim, gate, mps);
  }

  float wf[512];
  ss.ToWaveFunction(mps, wf);
  EXPECT_NEAR(wf[0], 0.0014289065729826689, 1e-4);
  EXPECT_NEAR(wf[1], -0.004883236717432737, 1e-4);
  EXPECT_NEAR(wf[2], 0.005564192309975624, 1e-4);
  EXPECT_NEAR(wf[3], -0.012465707957744598, 1e-4);
  EXPECT_NEAR(wf[4], -0.01509704627096653, 1e-4);
  EXPECT_NEAR(wf[5], 0.02856628969311714, 1e-4);
  EXPECT_NEAR(wf[6], -0.052923306822776794, 1e-4);
  EXPECT_NEAR(wf[7], 0.06542598456144333, 1e-4);
  EXPECT_NEAR(wf[8], 0.026189664378762245, 1e-4);
  EXPECT_NEAR(wf[9], 0.019128676503896713, 1e-4);
  EXPECT_NEAR(wf[10], 0.06991502642631531, 1e-4);
  EXPECT_NEAR(wf[11], 0.05682050436735153, 1e-4);
  EXPECT_NEAR(wf[12], 0.04927348345518112, 1e-4);
  EXPECT_NEAR(wf[13], 0.05772167444229126, 1e-4);
  EXPECT_NEAR(wf[14], 0.18450213968753815, 1e-4);
  EXPECT_NEAR(wf[15], 0.1270747184753418, 1e-4);
  EXPECT_NEAR(wf[16], 0.0004585272690746933, 1e-4);
  EXPECT_NEAR(wf[17], -0.0037279827520251274, 1e-4);
  EXPECT_NEAR(wf[18], 0.0025667455047369003, 1e-4);
  EXPECT_NEAR(wf[19], -0.009745213203132153, 1e-4);
  EXPECT_NEAR(wf[20], 0.0030728275887668133, 1e-4);
  EXPECT_NEAR(wf[21], -0.0027462595608085394, 1e-4);
  EXPECT_NEAR(wf[22], 0.006766265258193016, 1e-4);
  EXPECT_NEAR(wf[23], -0.009583592414855957, 1e-4);
  EXPECT_NEAR(wf[24], 0.003547749947756529, 1e-4);
  EXPECT_NEAR(wf[25], 0.0033205277286469936, 1e-4);
  EXPECT_NEAR(wf[26], 0.01306203380227089, 1e-4);
  EXPECT_NEAR(wf[27], 0.006699851714074612, 1e-4);
  EXPECT_NEAR(wf[28], -1.3373413821682334e-05, 1e-4);
  EXPECT_NEAR(wf[29], -0.0010807177750393748, 1e-4);
  EXPECT_NEAR(wf[30], -0.0008615776896476746, 1e-4);
  EXPECT_NEAR(wf[31], -0.0030719249043613672, 1e-4);
  EXPECT_NEAR(wf[32], -0.002316483296453953, 1e-4);
  EXPECT_NEAR(wf[33], 0.001494232565164566, 1e-4);
  EXPECT_NEAR(wf[34], -0.006698660086840391, 1e-4);
  EXPECT_NEAR(wf[35], 0.0031350301578640938, 1e-4);
  EXPECT_NEAR(wf[36], 0.002458656206727028, 1e-4);
  EXPECT_NEAR(wf[37], -0.012178290635347366, 1e-4);
  EXPECT_NEAR(wf[38], 0.01015305146574974, 1e-4);
  EXPECT_NEAR(wf[39], -0.028542086482048035, 1e-4);
  EXPECT_NEAR(wf[40], -0.008042107336223125, 1e-4);
  EXPECT_NEAR(wf[41], -0.007389616221189499, 1e-4);
  EXPECT_NEAR(wf[42], -0.022901810705661774, 1e-4);
  EXPECT_NEAR(wf[43], -0.023671913892030716, 1e-4);
  EXPECT_NEAR(wf[44], -0.02007935382425785, 1e-4);
  EXPECT_NEAR(wf[45], -0.017737379297614098, 1e-4);
  EXPECT_NEAR(wf[46], -0.0707787275314331, 1e-4);
  EXPECT_NEAR(wf[47], -0.035286612808704376, 1e-4);
  EXPECT_NEAR(wf[48], -0.0008655539713799953, 1e-4);
  EXPECT_NEAR(wf[49], 0.0010216656373813748, 1e-4);
  EXPECT_NEAR(wf[50], -0.002670466899871826, 1e-4);
  EXPECT_NEAR(wf[51], 0.0024032294750213623, 1e-4);
  EXPECT_NEAR(wf[52], -0.002122006379067898, 1e-4);
  EXPECT_NEAR(wf[53], -0.0015508199576288462, 1e-4);
  EXPECT_NEAR(wf[54], -0.005030060186982155, 1e-4);
  EXPECT_NEAR(wf[55], -0.003410058096051216, 1e-4);
  EXPECT_NEAR(wf[56], -0.0016508626285940409, 1e-4);
  EXPECT_NEAR(wf[57], -0.0012134818825870752, 1e-4);
  EXPECT_NEAR(wf[58], -0.006056999787688255, 1e-4);
  EXPECT_NEAR(wf[59], -0.0037506879307329655, 1e-4);
  EXPECT_NEAR(wf[60], -0.004103969316929579, 1e-4);
  EXPECT_NEAR(wf[61], 0.0004940991057083011, 1e-4);
  EXPECT_NEAR(wf[62], -0.01132766529917717, 1e-4);
  EXPECT_NEAR(wf[63], 0.004535992629826069, 1e-4);
  EXPECT_NEAR(wf[64], -0.014565523713827133, 1e-4);
  EXPECT_NEAR(wf[65], -0.009257279336452484, 1e-4);
  EXPECT_NEAR(wf[66], -0.03537633270025253, 1e-4);
  EXPECT_NEAR(wf[67], -0.029876481741666794, 1e-4);
  EXPECT_NEAR(wf[68], 0.07885720580816269, 1e-4);
  EXPECT_NEAR(wf[69], 0.0761096403002739, 1e-4);
  EXPECT_NEAR(wf[70], 0.16332173347473145, 1e-4);
  EXPECT_NEAR(wf[71], 0.2340961992740631, 1e-4);
  EXPECT_NEAR(wf[72], 0.08700724691152573, 1e-4);
  EXPECT_NEAR(wf[73], -0.06731436401605606, 1e-4);
  EXPECT_NEAR(wf[74], 0.25102484226226807, 1e-4);
  EXPECT_NEAR(wf[75], -0.17427709698677063, 1e-4);
  EXPECT_NEAR(wf[76], 0.23451192677021027, 1e-4);
  EXPECT_NEAR(wf[77], -0.10616815090179443, 1e-4);
  EXPECT_NEAR(wf[78], 0.5879150032997131, 1e-4);
  EXPECT_NEAR(wf[79], -0.481458842754364, 1e-4);
  EXPECT_NEAR(wf[80], -0.011161606758832932, 1e-4);
  EXPECT_NEAR(wf[81], -0.003173783188685775, 1e-4);
  EXPECT_NEAR(wf[82], -0.0285261869430542, 1e-4);
  EXPECT_NEAR(wf[83], -0.012472730129957199, 1e-4);
  EXPECT_NEAR(wf[84], -0.007169406395405531, 1e-4);
  EXPECT_NEAR(wf[85], -0.024708788841962814, 1e-4);
  EXPECT_NEAR(wf[86], -0.02162620984017849, 1e-4);
  EXPECT_NEAR(wf[87], -0.062582828104496, 1e-4);
  EXPECT_NEAR(wf[88], 0.0019742529839277267, 1e-4);
  EXPECT_NEAR(wf[89], -0.00907476432621479, 1e-4);
  EXPECT_NEAR(wf[90], 0.00041747279465198517, 1e-4);
  EXPECT_NEAR(wf[91], -0.039780646562576294, 1e-4);
  EXPECT_NEAR(wf[92], -0.030934825539588928, 1e-4);
  EXPECT_NEAR(wf[93], -0.009034967049956322, 1e-4);
  EXPECT_NEAR(wf[94], -0.09510712325572968, 1e-4);
  EXPECT_NEAR(wf[95], -0.0021972358226776123, 1e-4);
  EXPECT_NEAR(wf[96], -0.0006690436857752502, 1e-4);
  EXPECT_NEAR(wf[97], 0.009969948790967464, 1e-4);
  EXPECT_NEAR(wf[98], -0.005383004434406757, 1e-4);
  EXPECT_NEAR(wf[99], 0.026263613253831863, 1e-4);
  EXPECT_NEAR(wf[100], -0.015201564878225327, 1e-4);
  EXPECT_NEAR(wf[101], -0.021529488265514374, 1e-4);
  EXPECT_NEAR(wf[102], -0.025767244398593903, 1e-4);
  EXPECT_NEAR(wf[103], -0.05407162010669708, 1e-4);
  EXPECT_NEAR(wf[104], -0.030478909611701965, 1e-4);
  EXPECT_NEAR(wf[105], -0.0030686683021485806, 1e-4);
  EXPECT_NEAR(wf[106], -0.09224921464920044, 1e-4);
  EXPECT_NEAR(wf[107], -0.009406475350260735, 1e-4);
  EXPECT_NEAR(wf[108], -0.06196212023496628, 1e-4);
  EXPECT_NEAR(wf[109], -0.001441754400730133, 1e-4);
  EXPECT_NEAR(wf[110], -0.17780858278274536, 1e-4);
  EXPECT_NEAR(wf[111], 0.04309801757335663, 1e-4);
  EXPECT_NEAR(wf[112], -0.0008277512388303876, 1e-4);
  EXPECT_NEAR(wf[113], 0.002928556641563773, 1e-4);
  EXPECT_NEAR(wf[114], -0.003259341698139906, 1e-4);
  EXPECT_NEAR(wf[115], 0.007486439310014248, 1e-4);
  EXPECT_NEAR(wf[116], -0.005849980749189854, 1e-4);
  EXPECT_NEAR(wf[117], -0.001208490110002458, 1e-4);
  EXPECT_NEAR(wf[118], -0.01361632440239191, 1e-4);
  EXPECT_NEAR(wf[119], -0.0023143584839999676, 1e-4);
  EXPECT_NEAR(wf[120], -0.0045577045530080795, 1e-4);
  EXPECT_NEAR(wf[121], -0.0009546242654323578, 1e-4);
  EXPECT_NEAR(wf[122], -0.01606125570833683, 1e-4);
  EXPECT_NEAR(wf[123], -0.0020514349453151226, 1e-4);
  EXPECT_NEAR(wf[124], -0.008022841066122055, 1e-4);
  EXPECT_NEAR(wf[125], 0.004888217896223068, 1e-4);
  EXPECT_NEAR(wf[126], -0.019155969843268394, 1e-4);
  EXPECT_NEAR(wf[127], 0.020053446292877197, 1e-4);

  /* Equivalent Cirq code:

  import cirq

  def main():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)
    q3 = cirq.LineQubit(3)
    q4 = cirq.LineQubit(4)
    q5 = cirq.LineQubit(5)

    circuit = cirq.Circuit(
      cirq.Moment([
        cirq.Y(q0) ** 0.1,
        cirq.Y(q1) ** 0.2,
        cirq.Y(q2) ** 0.3,
        cirq.Y(q3) ** 0.4,
        cirq.Y(q4) ** 0.5,
        cirq.Y(q5) ** 0.6,
      ]),
      cirq.Moment([
        cirq.FSimGate(1.1,0.8)(q0, q1),
        cirq.FSimGate(0.2,0.9)(q2, q3),
        cirq.FSimGate(0.3,0.4)(q4, q5),
      ]),
      cirq.Moment([
        cirq.Y(q0) ** 0.88,
        cirq.Y(q1) ** 1.8,
        cirq.Y(q2) ** -0.3,
        cirq.Y(q3) ** 0.6,
        cirq.Y(q4) ** -1.2,
        cirq.Y(q5) ** 2.2,
      ]),
      cirq.Moment([
        cirq.FSimGate(0.5,-0.9)(q1, q2),
        cirq.FSimGate(-4.1,0.345)(q3, q4),
      ]),
    )

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    for i in range(len(result.state_vector())):
      wf2 = result.state_vector()
      print(f'EXPECT_NEAR(wf[{2*i}], {wf2[i].real}, 1e-4);')
      print(f'EXPECT_NEAR(wf[{2*i + 1}], {wf2[i].imag}, 1e-4);')

  if __name__ == '__main__':
    main()

  */
}

TEST(MPSSimulator, ApplyFusedGateLeft) {
  // Apply a fused gate matrix to the first two qubits.
  // Compute the state vector of:
  //   |     |     |
  // +-+-----+-+   |
  // |FusedGate|   |
  // +-+-----+-+   |
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto gate1 = GateCZ<float>::Create(2, 0, 1);
  auto gate2 = GateHd<float>::Create(0, 0);
  auto gate3 = GateHd<float>::Create(0, 1);

  GateFused<GateQSim<float>> fgate1{kGateCZ, 2, {0, 1}, &gate1,
                                    {&gate2, &gate3}, {}};
  CalculateFusedMatrix(fgate1);
  auto mps = ss.Create(3, 4);
  ss.SetStateZero(mps);
  ApplyFusedGate(sim, fgate1, mps);

  float wf[32];
  float ground_truth[] = {0.5, 0., 0., 0., 0.5, 0., 0., 0.,
                          0.5, 0., 0., 0., 0.5, 0., 0., 0.};
  ss.ToWaveFunction(mps, wf);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(wf[i], ground_truth[i], 1e-4);
  }
}

TEST(MPSSimulator, ApplyFusedGateRight) {
  // Apply a fused gate matrix to the last two qubits.
  // Compute the state vector of:
  //   |     |     |
  //   |   +-+-----+-+
  //   |   |FusedGate|
  //   |   +-+-----+-+ 
  //   |     |     |
  // +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |
  // +---+ +---+ +---+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto gate1 = GateCZ<float>::Create(2, 1, 2);
  auto gate2 = GateHd<float>::Create(0, 1);
  auto gate3 = GateHd<float>::Create(0, 2);

  GateFused<GateQSim<float>> fgate1{kGateCZ, 2, {1, 2}, &gate1,
                                    {&gate2, &gate3}, {}};
  CalculateFusedMatrix(fgate1);
  auto mps = ss.Create(3, 4);
  ss.SetStateZero(mps);
  ApplyFusedGate(sim, fgate1, mps);

  float wf[32];
  float ground_truth[] = {0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0.,
                          0., 0., 0., 0., 0., 0., 0., 0.};
  ss.ToWaveFunction(mps, wf);
  for (int i = 0; i < 16; i++) {
    EXPECT_NEAR(wf[i], ground_truth[i], 1e-4);
  }
}

TEST(MPSSimulator, ApplyFusedGateMiddle) {
  // Apply a fused gate matrix to the middle two qubits.
  // Compute the state vector of:
  //   |     |     |     |
  //   |   +-+-----+-+   |
  //   |   |FusedGate|   |
  //   |   +-+-----+-+   |
  //   |     |     |     |
  // +-+-+ +-+-+ +-+-+ +-+-+
  // | 0 +-+ 1 +-+ 2 |-| 3 |
  // +---+ +---+ +---+ +-+-+
  auto sim = MPSSimulator<For, float>(1);
  using MPSStateSpace = MPSSimulator<For, float>::MPSStateSpace_;
  auto ss = MPSStateSpace(1);

  auto gate1 = GateCZ<float>::Create(2, 1, 2);
  auto gate2 = GateHd<float>::Create(0, 1);
  auto gate3 = GateHd<float>::Create(0, 2);

  GateFused<GateQSim<float>> fgate1{kGateCZ, 2, {1, 2}, &gate1,
                                    {&gate2, &gate3}, {}};
  CalculateFusedMatrix(fgate1);
  auto mps = ss.Create(4, 4);
  ss.SetStateZero(mps);
  ApplyFusedGate(sim, fgate1, mps);

  float wf[64];
  float ground_truth[] = {0.5, 0., 0., 0., 0.5, 0., 0., 0.,
                          0.5, 0., 0., 0., 0.5, 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0.};
  ss.ToWaveFunction(mps, wf);
  for (int i = 0; i < 32; i++) {
    EXPECT_NEAR(wf[i], ground_truth[i], 1e-4);
  }
}

}  // namespace
}  // namespace mps
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
