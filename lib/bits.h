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

#ifndef BITS_H_
#define BITS_H_

#include <vector>

#ifdef __BMI2__

#include <immintrin.h>

#include <cstdint>

namespace qsim {
namespace bits {

inline uint32_t ExpandBits(uint32_t bits, unsigned n, uint32_t mask) {
  return _pdep_u32(bits, mask);
}

inline uint64_t ExpandBits(uint64_t bits, unsigned n, uint64_t mask) {
  return _pdep_u64(bits, mask);
}

inline uint32_t CompressBits(uint32_t bits, unsigned n, uint32_t mask) {
  return _pext_u32(bits, mask);
}

inline uint64_t CompressBits(uint64_t bits, unsigned n, uint64_t mask) {
  return _pext_u64(bits, mask);
}

}  // namespace bits
}  // namespace qsim

#else  // __BMI2__

namespace qsim {
namespace bits {

template <typename Integer>
inline Integer ExpandBits(Integer bits, unsigned n, Integer mask) {
  Integer ebits = 0;
  unsigned k = 0;

  for (unsigned i = 0; i < n; ++i) {
    if ((mask >> i) & 1) {
      ebits |= ((bits >> k) & 1) << i;
      ++k;
    }
  }

  return ebits;
}

template <typename Integer>
inline Integer CompressBits(Integer bits, unsigned n, Integer mask) {
  Integer sbits = 0;
  unsigned k = 0;

  for (unsigned i = 0; i < n; ++i) {
    if ((mask >> i) & 1) {
      sbits |= ((bits >> i) & 1) << k;
      ++k;
    }
  }

  return sbits;
}

}  // namespace bits
}  // namespace qsim

#endif  // __BMI2__

namespace qsim {
namespace bits {

template <typename Integer>
inline Integer PermuteBits(
    Integer bits, unsigned n, const std::vector<unsigned>& perm) {
  Integer pbits = 0;

  for (unsigned i = 0; i < n; ++i) {
    pbits |= ((bits >> i) & 1) << perm[i];
  }

  return pbits;
}

}  // namespace bits
}  // namespace qsim

#endif  // BITS_H_
