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

#ifndef UTIL_H_
#define UTIL_H_

#include <time.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace qsim {

template <typename Container>
inline void SplitString(
    const std::string& str, char delim, Container& words) {
  words.resize(0);

  std::string word;
  std::stringstream ss(str);

  while (std::getline(ss, word, delim)) {
    words.push_back(std::move(word));
  }
}

template <typename Op, typename Container>
inline void SplitString(
    const std::string& str, char delim, Op op, Container& words) {
  words.resize(0);

  std::string word;
  std::stringstream ss(str);

  while (std::getline(ss, word, delim)) {
    words.push_back(op(word));
  }
}

inline double GetTime() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return time.tv_sec + 1e-9 * time.tv_nsec;
}

}  // namespace qsim

#endif  // UTIL_H_
