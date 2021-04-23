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

#ifndef CPUINFO_H_
#define CPUINFO_H_

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace qsim {

inline bool HaveAVX512() {
  unsigned info[4];

#ifdef _WIN32
  __cpuidex(info, 7, 0);
#else
  if (__get_cpuid_count(7, 0, info, info + 1, info + 2, info + 3) == 0) {
    info[1] = 0;
  }
#endif

  return (info[1] & (unsigned{1} << 16)) != 0;
}

}  // namespace qsim

#endif  // CPUINFO_H_
