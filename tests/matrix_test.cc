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

TEST(MatrixTest, Matrix2Multiply) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8> m{8, 7, 6, 5, 4, 3, 2, 1};

  Matrix2Multiply(u, m);

  EXPECT_FLOAT_EQ(m[0], -6);
  EXPECT_FLOAT_EQ(m[1], 48);
  EXPECT_FLOAT_EQ(m[2], -2);
  EXPECT_FLOAT_EQ(m[3], 28);
  EXPECT_FLOAT_EQ(m[4], 2);
  EXPECT_FLOAT_EQ(m[5], 136);
  EXPECT_FLOAT_EQ(m[6], 6);
  EXPECT_FLOAT_EQ(m[7], 84);
}

TEST(MatrixTest, Matrix2ScalarMultiply) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};

  Matrix2ScalarMultiply(3, u);
  for (unsigned i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(u[i], (i + 1) * 3);
  }
}

TEST(MatrixTest, Matrix2Dagger) {
  std::array<float, 8> u{0, 1, 2, 3, 4, 5, 6, 7};
  Matrix2Dagger(u);
  EXPECT_FLOAT_EQ(u[0], 0);
  EXPECT_FLOAT_EQ(u[1], -1);
  EXPECT_FLOAT_EQ(u[2], 4);
  EXPECT_FLOAT_EQ(u[3], -5);
  EXPECT_FLOAT_EQ(u[4], 2);
  EXPECT_FLOAT_EQ(u[5], -3);
  EXPECT_FLOAT_EQ(u[6], 6);
  EXPECT_FLOAT_EQ(u[7], -7);
}

TEST(MatrixTest, Matrix4Multiply20) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 32> m{32, 31, 30, 29, 28, 27, 26, 25,
                          24, 23, 22, 21, 20, 19, 18, 17,
                          16, 15, 14, 13, 12, 11, 10, 9,
                          8, 7, 6, 5, 4, 3, 2, 1};

  Matrix4Multiply20(u, m);

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
}

TEST(MatrixTest, Matrix4Multiply21) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 32> m{32, 31, 30, 29, 28, 27, 26, 25,
                          24, 23, 22, 21, 20, 19, 18, 17,
                          16, 15, 14, 13, 12, 11, 10, 9,
                          8, 7, 6, 5, 4, 3, 2, 1};

  Matrix4Multiply21(u, m);

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

TEST(MatrixTest, Matrix4Multiply) {
  std::array<float, 32> u{1, 2, 3, 4, 5, 6, 7, 8,
                          9, 10, 11, 12, 13, 14, 15, 16,
                          17, 18, 19, 20, 21, 22, 23, 24,
                          25, 26, 27, 28, 29, 30, 31, 32};
  std::array<float, 32> m{32, 31, 30, 29, 28, 27, 26, 25,
                          24, 23, 22, 21, 20, 19, 18, 17,
                          16, 15, 14, 13, 12, 11, 10, 9,
                          8, 7, 6, 5, 4, 3, 2, 1};
  Matrix4Multiply(u, m);

  EXPECT_FLOAT_EQ(m[0], -60);
  EXPECT_FLOAT_EQ(m[1], 544);
  EXPECT_FLOAT_EQ(m[2], -52);
  EXPECT_FLOAT_EQ(m[3], 472);
  EXPECT_FLOAT_EQ(m[4], -44);
  EXPECT_FLOAT_EQ(m[5], 400);
  EXPECT_FLOAT_EQ(m[6], -36);
  EXPECT_FLOAT_EQ(m[7], 328);
  EXPECT_FLOAT_EQ(m[8], -28);
  EXPECT_FLOAT_EQ(m[9], 1792);
  EXPECT_FLOAT_EQ(m[10], -20);
  EXPECT_FLOAT_EQ(m[11], 1592);
  EXPECT_FLOAT_EQ(m[12], -12);
  EXPECT_FLOAT_EQ(m[13], 1392);
  EXPECT_FLOAT_EQ(m[14], -4);
  EXPECT_FLOAT_EQ(m[15], 1192);
  EXPECT_FLOAT_EQ(m[16], 4);
  EXPECT_FLOAT_EQ(m[17], 3040);
  EXPECT_FLOAT_EQ(m[18], 12);
  EXPECT_FLOAT_EQ(m[19], 2712);
  EXPECT_FLOAT_EQ(m[20], 20);
  EXPECT_FLOAT_EQ(m[21], 2384);
  EXPECT_FLOAT_EQ(m[22], 28);
  EXPECT_FLOAT_EQ(m[23], 2056);
  EXPECT_FLOAT_EQ(m[24], 36);
  EXPECT_FLOAT_EQ(m[25], 4288);
  EXPECT_FLOAT_EQ(m[26], 44);
  EXPECT_FLOAT_EQ(m[27], 3832);
  EXPECT_FLOAT_EQ(m[28], 52);
  EXPECT_FLOAT_EQ(m[29], 3376);
  EXPECT_FLOAT_EQ(m[30], 60);
  EXPECT_FLOAT_EQ(m[31], 2920);
}

TEST(MatrixTest, Matrix4Permute) {
  // Conjugation by swap gate:
  //  | 0  1  2  3  |      | 0  2  1  3  |
  //  | 4  5  6  7  |      | 8  10 9  11 |
  //  | 8  9  10 11 | ---> | 4  6  5  7  |
  //  | 12 13 14 15 |      | 12 14 13 15 |
  // clang-format off
  std::array<float, 32> matrix{
    0,  0.5, 1, 1.5, 2, 2.5, 3, 3.5,
    4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
    8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
    12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5};
  const std::array<float, 32> matrix_swapped{
    0,  0.5, 2, 2.5, 1, 1.5, 3, 3.5,
    8, 8.5, 10, 10.5, 9, 9.5, 11, 11.5,
    4, 4.5, 6, 6.5, 5, 5.5, 7, 7.5,
    12, 12.5, 14, 14.5, 13, 13.5, 15, 15.5};
  // clang-format on
  Matrix4Permute(matrix);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(matrix[i], matrix_swapped[i]);
  }
}

TEST(MatrixTest, Matrix4ScalarMultiply) {
  std::array<float, 32> u{1, 2, 3, 4, 5, 6, 7, 8,
                          9, 10, 11, 12, 13, 14, 15, 16,
                          17, 18, 19, 20, 21, 22, 23, 24,
                          25, 26, 27, 28, 29, 30, 31, 32};

  Matrix4ScalarMultiply(3, u);
  for (unsigned i = 0; i < 32; i++) {
    EXPECT_FLOAT_EQ(u[i], (i + 1) * 3);
  }
}

TEST(MatrixTest, Matrix4Dagger) {
  std::array<float, 32> u{1, 2, 3, 4, 5, 6, 7, 8,
                          9, 10, 11, 12, 13, 14, 15, 16,
                          17, 18, 19, 20, 21, 22, 23, 24,
                          25, 26, 27, 28, 29, 30, 31, 32};
  Matrix4Dagger(u);
  EXPECT_FLOAT_EQ(u[0], 1);
  EXPECT_FLOAT_EQ(u[1], -2);
  EXPECT_FLOAT_EQ(u[2], 9);
  EXPECT_FLOAT_EQ(u[3], -10);
  EXPECT_FLOAT_EQ(u[4], 17);
  EXPECT_FLOAT_EQ(u[5], -18);
  EXPECT_FLOAT_EQ(u[6], 25);
  EXPECT_FLOAT_EQ(u[7], -26);
  EXPECT_FLOAT_EQ(u[8], 3);
  EXPECT_FLOAT_EQ(u[9], -4);
  EXPECT_FLOAT_EQ(u[10], 11);
  EXPECT_FLOAT_EQ(u[11], -12);
  EXPECT_FLOAT_EQ(u[12], 19);
  EXPECT_FLOAT_EQ(u[13], -20);
  EXPECT_FLOAT_EQ(u[14], 27);
  EXPECT_FLOAT_EQ(u[15], -28);
  EXPECT_FLOAT_EQ(u[16], 5);
  EXPECT_FLOAT_EQ(u[17], -6);
  EXPECT_FLOAT_EQ(u[18], 13);
  EXPECT_FLOAT_EQ(u[19], -14);
  EXPECT_FLOAT_EQ(u[20], 21);
  EXPECT_FLOAT_EQ(u[21], -22);
  EXPECT_FLOAT_EQ(u[22], 29);
  EXPECT_FLOAT_EQ(u[23], -30);
  EXPECT_FLOAT_EQ(u[24], 7);
  EXPECT_FLOAT_EQ(u[25], -8);
  EXPECT_FLOAT_EQ(u[26], 15);
  EXPECT_FLOAT_EQ(u[27], -16);
  EXPECT_FLOAT_EQ(u[28], 23);
  EXPECT_FLOAT_EQ(u[29], -24);
  EXPECT_FLOAT_EQ(u[30], 31);
  EXPECT_FLOAT_EQ(u[31], -32);
}
  
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
