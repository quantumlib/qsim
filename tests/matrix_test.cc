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

#include <array>

#include "gtest/gtest.h"

#include "../lib/matrix.h"

namespace qsim {

TEST(MatrixTest, MatrixMultiply1) {
  Matrix<float> u1 = {1, 2, 3, 4, 5, 6, 7, 8};
  Matrix<float> m1 = {8, 7, 6, 5, 4, 3, 2, 1};

  MatrixMultiply(1, u1, m1);

  EXPECT_FLOAT_EQ(m1[0], -6);
  EXPECT_FLOAT_EQ(m1[1], 48);
  EXPECT_FLOAT_EQ(m1[2], -2);
  EXPECT_FLOAT_EQ(m1[3], 28);
  EXPECT_FLOAT_EQ(m1[4], 2);
  EXPECT_FLOAT_EQ(m1[5], 136);
  EXPECT_FLOAT_EQ(m1[6], 6);
  EXPECT_FLOAT_EQ(m1[7], 84);

  Matrix<float> u2 = { 1,  2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      25, 26, 27, 28, 29, 30, 31, 32};
  Matrix<float> m2 = {32, 31, 30, 29, 28, 27, 26, 25,
                      24, 23, 22, 21, 20, 19, 18, 17,
                      16, 15, 14, 13, 12, 11, 10,  9,
                       8,  7,  6,  5,  4,  3,  2,  1};

  MatrixMultiply(2, u2, m2);

  EXPECT_FLOAT_EQ(m2[0], -60);
  EXPECT_FLOAT_EQ(m2[1], 544);
  EXPECT_FLOAT_EQ(m2[2], -52);
  EXPECT_FLOAT_EQ(m2[3], 472);
  EXPECT_FLOAT_EQ(m2[4], -44);
  EXPECT_FLOAT_EQ(m2[5], 400);
  EXPECT_FLOAT_EQ(m2[6], -36);
  EXPECT_FLOAT_EQ(m2[7], 328);
  EXPECT_FLOAT_EQ(m2[8], -28);
  EXPECT_FLOAT_EQ(m2[9], 1792);
  EXPECT_FLOAT_EQ(m2[10], -20);
  EXPECT_FLOAT_EQ(m2[11], 1592);
  EXPECT_FLOAT_EQ(m2[12], -12);
  EXPECT_FLOAT_EQ(m2[13], 1392);
  EXPECT_FLOAT_EQ(m2[14], -4);
  EXPECT_FLOAT_EQ(m2[15], 1192);
  EXPECT_FLOAT_EQ(m2[16], 4);
  EXPECT_FLOAT_EQ(m2[17], 3040);
  EXPECT_FLOAT_EQ(m2[18], 12);
  EXPECT_FLOAT_EQ(m2[19], 2712);
  EXPECT_FLOAT_EQ(m2[20], 20);
  EXPECT_FLOAT_EQ(m2[21], 2384);
  EXPECT_FLOAT_EQ(m2[22], 28);
  EXPECT_FLOAT_EQ(m2[23], 2056);
  EXPECT_FLOAT_EQ(m2[24], 36);
  EXPECT_FLOAT_EQ(m2[25], 4288);
  EXPECT_FLOAT_EQ(m2[26], 44);
  EXPECT_FLOAT_EQ(m2[27], 3832);
  EXPECT_FLOAT_EQ(m2[28], 52);
  EXPECT_FLOAT_EQ(m2[29], 3376);
  EXPECT_FLOAT_EQ(m2[30], 60);
  EXPECT_FLOAT_EQ(m2[31], 2920);
}

TEST(MatrixTest, MatrixMultiply2) {
  Matrix<float> u = {1, 2, 3, 4, 5, 6, 7, 8};
  Matrix<float> m0 = {32, 31, 30, 29, 28, 27, 26, 25,
                      24, 23, 22, 21, 20, 19, 18, 17,
                      16, 15, 14, 13, 12, 11, 10,  9,
                       8,  7,  6,  5,  4,  3,  2,  1};

  auto m = m0;
  MatrixMultiply(1, 1, u, 2, m);

  EXPECT_FLOAT_EQ(m[0], -50);
  EXPECT_FLOAT_EQ(m[1], 260);
  EXPECT_FLOAT_EQ(m[2], -46);
  EXPECT_FLOAT_EQ(m[3], 240);
  EXPECT_FLOAT_EQ(m[4], -42);
  EXPECT_FLOAT_EQ(m[5], 220);
  EXPECT_FLOAT_EQ(m[6], -38);
  EXPECT_FLOAT_EQ(m[7], 200);
  EXPECT_FLOAT_EQ(m[8], -42);
  EXPECT_FLOAT_EQ(m[9], 700);
  EXPECT_FLOAT_EQ(m[10], -38);
  EXPECT_FLOAT_EQ(m[11], 648);
  EXPECT_FLOAT_EQ(m[12], -34);
  EXPECT_FLOAT_EQ(m[13], 596);
  EXPECT_FLOAT_EQ(m[14], -30);
  EXPECT_FLOAT_EQ(m[15], 544);
  EXPECT_FLOAT_EQ(m[16], -18);
  EXPECT_FLOAT_EQ(m[17], 100);
  EXPECT_FLOAT_EQ(m[18], -14);
  EXPECT_FLOAT_EQ(m[19], 80);
  EXPECT_FLOAT_EQ(m[20], -10);
  EXPECT_FLOAT_EQ(m[21], 60);
  EXPECT_FLOAT_EQ(m[22], -6);
  EXPECT_FLOAT_EQ(m[23], 40);
  EXPECT_FLOAT_EQ(m[24], -10);
  EXPECT_FLOAT_EQ(m[25], 284);
  EXPECT_FLOAT_EQ(m[26], -6);
  EXPECT_FLOAT_EQ(m[27], 232);
  EXPECT_FLOAT_EQ(m[28], -2);
  EXPECT_FLOAT_EQ(m[29], 180);
  EXPECT_FLOAT_EQ(m[30], 2);
  EXPECT_FLOAT_EQ(m[31], 128);

  m = m0;
  MatrixMultiply(2, 1, u, 2, m);

  EXPECT_FLOAT_EQ(m[0], -42);
  EXPECT_FLOAT_EQ(m[1], 204);
  EXPECT_FLOAT_EQ(m[2], -38);
  EXPECT_FLOAT_EQ(m[3], 184);
  EXPECT_FLOAT_EQ(m[4], -34);
  EXPECT_FLOAT_EQ(m[5], 164);
  EXPECT_FLOAT_EQ(m[6], -30);
  EXPECT_FLOAT_EQ(m[7], 144);
  EXPECT_FLOAT_EQ(m[8], -26);
  EXPECT_FLOAT_EQ(m[9], 124);
  EXPECT_FLOAT_EQ(m[10], -22);
  EXPECT_FLOAT_EQ(m[11], 104);
  EXPECT_FLOAT_EQ(m[12], -18);
  EXPECT_FLOAT_EQ(m[13], 84);
  EXPECT_FLOAT_EQ(m[14], -14);
  EXPECT_FLOAT_EQ(m[15], 64);
  EXPECT_FLOAT_EQ(m[16], -34);
  EXPECT_FLOAT_EQ(m[17], 580);
  EXPECT_FLOAT_EQ(m[18], -30);
  EXPECT_FLOAT_EQ(m[19], 528);
  EXPECT_FLOAT_EQ(m[20], -26);
  EXPECT_FLOAT_EQ(m[21], 476);
  EXPECT_FLOAT_EQ(m[22], -22);
  EXPECT_FLOAT_EQ(m[23], 424);
  EXPECT_FLOAT_EQ(m[24], -18);
  EXPECT_FLOAT_EQ(m[25], 372);
  EXPECT_FLOAT_EQ(m[26], -14);
  EXPECT_FLOAT_EQ(m[27], 320);
  EXPECT_FLOAT_EQ(m[28], -10);
  EXPECT_FLOAT_EQ(m[29], 268);
  EXPECT_FLOAT_EQ(m[30], -6);
  EXPECT_FLOAT_EQ(m[31], 216);
}

TEST(MatrixTest, MatrixScalarMultiply) {
  Matrix<float>  m1 = {1, 2, 3, 4, 5, 6, 7, 8};

  MatrixScalarMultiply(3, m1);

  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(m1[i], (i + 1) * 3);
  }

  Matrix<float> m2 = { 1,  2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      25, 26, 27, 28, 29, 30, 31, 32};

  MatrixScalarMultiply(3, m2);

  for (unsigned i = 0; i < 32; ++i) {
    EXPECT_FLOAT_EQ(m2[i], (i + 1) * 3);
  }
}

TEST(MatrixTest, MatrixDagger) {
  Matrix<float> m1 = {0, 1, 2, 3, 4, 5, 6, 7};

  MatrixDagger(2, m1);

  EXPECT_FLOAT_EQ(m1[0], 0);
  EXPECT_FLOAT_EQ(m1[1], -1);
  EXPECT_FLOAT_EQ(m1[2], 4);
  EXPECT_FLOAT_EQ(m1[3], -5);
  EXPECT_FLOAT_EQ(m1[4], 2);
  EXPECT_FLOAT_EQ(m1[5], -3);
  EXPECT_FLOAT_EQ(m1[6], 6);
  EXPECT_FLOAT_EQ(m1[7], -7);

  Matrix<float> m2 = { 1,  2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      25, 26, 27, 28, 29, 30, 31, 32};

  MatrixDagger(4, m2);

  EXPECT_FLOAT_EQ(m2[0], 1);
  EXPECT_FLOAT_EQ(m2[1], -2);
  EXPECT_FLOAT_EQ(m2[2], 9);
  EXPECT_FLOAT_EQ(m2[3], -10);
  EXPECT_FLOAT_EQ(m2[4], 17);
  EXPECT_FLOAT_EQ(m2[5], -18);
  EXPECT_FLOAT_EQ(m2[6], 25);
  EXPECT_FLOAT_EQ(m2[7], -26);
  EXPECT_FLOAT_EQ(m2[8], 3);
  EXPECT_FLOAT_EQ(m2[9], -4);
  EXPECT_FLOAT_EQ(m2[10], 11);
  EXPECT_FLOAT_EQ(m2[11], -12);
  EXPECT_FLOAT_EQ(m2[12], 19);
  EXPECT_FLOAT_EQ(m2[13], -20);
  EXPECT_FLOAT_EQ(m2[14], 27);
  EXPECT_FLOAT_EQ(m2[15], -28);
  EXPECT_FLOAT_EQ(m2[16], 5);
  EXPECT_FLOAT_EQ(m2[17], -6);
  EXPECT_FLOAT_EQ(m2[18], 13);
  EXPECT_FLOAT_EQ(m2[19], -14);
  EXPECT_FLOAT_EQ(m2[20], 21);
  EXPECT_FLOAT_EQ(m2[21], -22);
  EXPECT_FLOAT_EQ(m2[22], 29);
  EXPECT_FLOAT_EQ(m2[23], -30);
  EXPECT_FLOAT_EQ(m2[24], 7);
  EXPECT_FLOAT_EQ(m2[25], -8);
  EXPECT_FLOAT_EQ(m2[26], 15);
  EXPECT_FLOAT_EQ(m2[27], -16);
  EXPECT_FLOAT_EQ(m2[28], 23);
  EXPECT_FLOAT_EQ(m2[29], -24);
  EXPECT_FLOAT_EQ(m2[30], 31);
  EXPECT_FLOAT_EQ(m2[31], -32);
}

TEST(MatrixTest, MatrixShuffle) {
  Matrix<float> sw = {1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 0};

  Matrix<float> m2 = { 1,  2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      25, 26, 27, 28, 29, 30, 31, 32};

  auto v2 = m2;
  auto perm2 = NormalToGateOrderPermutation({7, 3});
  MatrixShuffle(perm2, 2, v2);

  // v2 should be the same as sw * m2 * sw.

  auto s2 = sw;
  MatrixMultiply(2, m2, s2);
  MatrixMultiply(2, sw, s2);

  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(v2[i], s2[i]);
  }

  Matrix<float> m3(2 * 8 * 8);
  for (std::size_t i = 0; i < m3.size(); ++i) {
    m3[i] = i + 1;
  }

  auto v3 = m3;
  auto perm3 = NormalToGateOrderPermutation({7, 1, 3});
  MatrixShuffle(perm3, 3, v3);

  // {1, 3, 7} -> {7, 1, 3}.
  // v3 should be the same as sw(0, 2) * sw(1, 2) * m3 * sw(1, 2) * sw(0, 2).
  // sw(q1, s2) is a swap matrix acting on qubits q1 and q2.

  Matrix<float> s3;
  MatrixIdentity(8, s3);
  MatrixMultiply(5, 2, sw, 3, s3);
  MatrixMultiply(6, 2, sw, 3, s3);
  MatrixMultiply(3, m3, s3);
  MatrixMultiply(6, 2, sw, 3, s3);
  MatrixMultiply(5, 2, sw, 3, s3);

  for (int i = 0; i < 128; i++) {
    EXPECT_EQ(v3[i], s3[i]);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
