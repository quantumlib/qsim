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

#ifndef UTIL_CPU_H_
#define UTIL_CPU_H_

#ifdef __SSE2__
# include <immintrin.h>
#endif

namespace qsim {

// This function sets flush-to-zero and denormals-are-zeros MXCSR control
// flags. This prevents rare cases of performance slowdown potentially at
// the cost of a tiny precision loss.
inline void SetFlushToZeroAndDenormalsAreZeros() {
#ifdef __SSE2__
  _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
}

// This function clears flush-to-zero and denormals-are-zeros MXCSR control
// flags.
inline void ClearFlushToZeroAndDenormalsAreZeros() {
#ifdef __SSE2__
  _mm_setcsr(_mm_getcsr() & ~unsigned{0x8040});
#endif
}

}  // namespace qsim

#endif  // UTIL_CPU_H_
