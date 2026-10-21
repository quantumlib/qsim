// Copyright 2026 Google LLC. All Rights Reserved.
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

#include <stdexcept>

#ifdef __SSE2__
#include <immintrin.h>
#endif

#include "gtest/gtest.h"

#include "../lib/util_cpu.h"

namespace qsim {
namespace {

#ifdef __SSE2__

constexpr unsigned kFlushToZeroAndDenormalsAreZeros = 0x8040;

TEST(ScopedFlushToZeroAndDenormalsAreZerosTest, EnforcesAndRestoresFlags) {
  const unsigned original_flags = _mm_getcsr();

  _mm_setcsr(original_flags & ~kFlushToZeroAndDenormalsAreZeros);
  {
    ScopedFlushToZeroAndDenormalsAreZeros guard(true);
    EXPECT_EQ(_mm_getcsr(), original_flags | kFlushToZeroAndDenormalsAreZeros);
  }
  EXPECT_EQ(_mm_getcsr(), original_flags & ~kFlushToZeroAndDenormalsAreZeros);

  _mm_setcsr(original_flags | kFlushToZeroAndDenormalsAreZeros);
  {
    ScopedFlushToZeroAndDenormalsAreZeros guard(false);
    EXPECT_EQ(_mm_getcsr(), original_flags & ~kFlushToZeroAndDenormalsAreZeros);
  }
  EXPECT_EQ(_mm_getcsr(), original_flags | kFlushToZeroAndDenormalsAreZeros);

  _mm_setcsr(original_flags);
}

TEST(ScopedFlushToZeroAndDenormalsAreZerosTest, RestoresNestedGuards) {
  const unsigned original_flags = _mm_getcsr();

  {
    ScopedFlushToZeroAndDenormalsAreZeros outer;
    const unsigned outer_flags = _mm_getcsr();
    EXPECT_EQ(outer_flags, original_flags | kFlushToZeroAndDenormalsAreZeros);

    {
      ScopedFlushToZeroAndDenormalsAreZeros inner;
      EXPECT_EQ(_mm_getcsr(), outer_flags);
    }

    EXPECT_EQ(_mm_getcsr(), outer_flags);
  }

  EXPECT_EQ(_mm_getcsr(), original_flags);
}

TEST(ScopedFlushToZeroAndDenormalsAreZerosTest, RestoresFlagsDuringException) {
  const unsigned original_flags = _mm_getcsr();

  try {
    ScopedFlushToZeroAndDenormalsAreZeros guard;
    EXPECT_EQ(_mm_getcsr(), original_flags | kFlushToZeroAndDenormalsAreZeros);
    throw std::runtime_error("test exception");
  } catch (const std::runtime_error&) {
  }

  EXPECT_EQ(_mm_getcsr(), original_flags);
}

#else

TEST(ScopedFlushToZeroAndDenormalsAreZerosTest, RequiresSse2) {
  GTEST_SKIP() << "MXCSR is only available with SSE2";
}

#endif

}  // namespace
}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
