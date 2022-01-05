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

#include <algorithm>
#include <chrono>
#include <random>
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
  using namespace std::chrono;
  steady_clock::duration since_epoch = steady_clock::now().time_since_epoch();
  return double(since_epoch.count() * steady_clock::period::num)
                                    / steady_clock::period::den;
}

template <typename DistrRealType, typename RGen>
inline DistrRealType RandomValue(RGen& rgen, DistrRealType max_value) {
  std::uniform_real_distribution<DistrRealType> distr(0.0, max_value);
  return distr(rgen);
}

template <typename DistrRealType>
inline std::vector<DistrRealType> GenerateRandomValues(
    uint64_t num_samples, unsigned seed, DistrRealType max_value) {
  std::vector<DistrRealType> rs;
  rs.reserve(num_samples + 1);

  std::mt19937 rgen(seed);
  std::uniform_real_distribution<DistrRealType> distr(0.0, max_value);

  for (uint64_t i = 0; i < num_samples; ++i) {
    rs.emplace_back(distr(rgen));
  }

  std::sort(rs.begin(), rs.end());
  // Populate the final element to prevent sanitizer errors.
  rs.emplace_back(max_value);

  return rs;
}

}  // namespace qsim

#endif  // UTIL_H_
