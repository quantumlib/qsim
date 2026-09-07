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

// Gets the current MXCSR control flags.
inline unsigned GetMXCSRControlFlags() {
#ifdef __SSE2__
  return _mm_getcsr();
#else
  return 0;
#endif
}

// Sets the MXCSR control flags to a specific value.
inline void SetMXCSRControlFlags(unsigned value) {
#ifdef __SSE2__
  _mm_setcsr(value);
#endif
}

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

// RAII guard for the flush-to-zero and denormals-are-zeros MXCSR control flags.
class ScopedFlushToZeroAndDenormalsAreZeros {
 public:
  explicit ScopedFlushToZeroAndDenormalsAreZeros(bool denormals_are_zeros)
      : original_flags_(GetMXCSRControlFlags()) {
    if (denormals_are_zeros) {
      SetFlushToZeroAndDenormalsAreZeros();
    } else {
      ClearFlushToZeroAndDenormalsAreZeros();
    }
  }

  ~ScopedFlushToZeroAndDenormalsAreZeros() {
    SetMXCSRControlFlags(original_flags_);
  }

  // Prevent copying.
  ScopedFlushToZeroAndDenormalsAreZeros(
      const ScopedFlushToZeroAndDenormalsAreZeros&) = delete;
  ScopedFlushToZeroAndDenormalsAreZeros& operator=(
      const ScopedFlushToZeroAndDenormalsAreZeros&) = delete;

  // Prevent moving
  ScopedFlushToZeroAndDenormalsAreZeros(
      ScopedFlushToZeroAndDenormalsAreZeros&&) = delete;
  ScopedFlushToZeroAndDenormalsAreZeros& operator=(
      ScopedFlushToZeroAndDenormalsAreZeros&&) = delete;

 private:
  unsigned original_flags_;
};

}  // namespace qsim

#endif  // UTIL_CPU_H_
