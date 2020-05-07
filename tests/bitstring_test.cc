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

#include <sstream>

#include "gtest/gtest.h"

#include "../lib/bitstring.h"

namespace qsim {

struct IO {
  static void errorf(const char* format, ...) {}
  static void messagef(const char* format, ...) {}
};

constexpr char provider[] = "bitstring_test";

TEST(BitstringTest, ValidBitstrings) {
  constexpr char valid_bitstrings[] =
R"(1000000
0100000
0100100
1001010
1000101
1110000
)";

  std::stringstream ss(valid_bitstrings);
  std::vector<Bitstring> bitstrings;

  EXPECT_EQ(BitstringsFromStream<IO>(7, provider, ss, bitstrings), true);
  EXPECT_EQ(bitstrings.size(), 6);
  EXPECT_EQ(bitstrings[0], 1);
  EXPECT_EQ(bitstrings[1], 2);
  EXPECT_EQ(bitstrings[2], 18);
  EXPECT_EQ(bitstrings[3], 41);
  EXPECT_EQ(bitstrings[4], 81);
  EXPECT_EQ(bitstrings[5], 7);
}

TEST(BitstringTest, InValidBitstrings1) {
  constexpr char invalid_bitstrings[] =
R"(1000000
0100000
010010
1001010
1000101
1110000
)";

  std::stringstream ss(invalid_bitstrings);
  std::vector<Bitstring> bitstrings;

  EXPECT_EQ(BitstringsFromStream<IO>(7, provider, ss, bitstrings), false);
  EXPECT_EQ(bitstrings.size(), 0);
}

TEST(BitstringTest, InValidBitstrings2) {
  constexpr char invalid_bitstrings[] =
R"(1000000
0100000
0100100
1001010
10001011
1110000
)";

  std::stringstream ss(invalid_bitstrings);
  std::vector<Bitstring> bitstrings;

  EXPECT_EQ(BitstringsFromStream<IO>(7, provider, ss, bitstrings), false);
  EXPECT_EQ(bitstrings.size(), 0);
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
