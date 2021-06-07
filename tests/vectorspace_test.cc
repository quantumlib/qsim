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

#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/vectorspace.h"

namespace qsim {

TEST(VectorSpaceTest, BasicTest) {
  struct DummyImplementation {
    static uint64_t MinSize(unsigned num_qubits) {
      return uint64_t{1} << num_qubits;
    }
  };

  unsigned num_qubits = 4;
  uint64_t size = uint64_t{1} << num_qubits;

  VectorSpace<DummyImplementation, For, float> vector_space(1);

  auto vector1 = vector_space.Create(num_qubits);

  EXPECT_FALSE(vector_space.IsNull(vector1));
  EXPECT_NE(vector1.get(), nullptr);
  EXPECT_EQ(uint64_t(vector1.get()) % 64, 0);
  EXPECT_EQ(vector1.num_qubits(), num_qubits);

  std::vector<float> buf(size, 0);

  auto vector2 = vector_space.Create(buf.data(), num_qubits);
  EXPECT_FALSE(vector_space.IsNull(vector2));
  EXPECT_EQ(vector2.get(), buf.data());
  EXPECT_EQ(vector2.num_qubits(), num_qubits);

  for (uint64_t i = 0; i < size; ++i) {
    vector1.get()[i] = i + 1;
  }

  vector_space.Copy(vector1, vector2);

  for (uint64_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(vector2.get()[i], i + 1);
  }

  auto vector3 = vector_space.Null();
  EXPECT_TRUE(vector_space.IsNull(vector3));
  EXPECT_EQ(vector3.get(), nullptr);
  EXPECT_EQ(vector3.num_qubits(), 0);

  auto p1 = vector1.get();

  std::swap(vector1, vector3);

  EXPECT_TRUE(vector_space.IsNull(vector1));
  EXPECT_EQ(vector1.get(), nullptr);

  EXPECT_FALSE(vector_space.IsNull(vector3));
  EXPECT_EQ(vector3.get(), p1);
  EXPECT_EQ(vector3.num_qubits(), num_qubits);

  for (uint64_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(vector3.get()[i], i + 1);
  }

  auto p2 = vector2.release();

  EXPECT_TRUE(vector_space.IsNull(vector2));
  EXPECT_EQ(vector2.get(), nullptr);
  EXPECT_EQ(vector2.num_qubits(), 0);
  EXPECT_EQ(p2, buf.data());

  auto p3 = vector3.release();

  EXPECT_TRUE(vector_space.IsNull(vector3));
  EXPECT_EQ(vector3.get(), nullptr);
  EXPECT_EQ(vector3.num_qubits(), 0);
  EXPECT_EQ(p3, p1);

  vector_space.Free(p3);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
