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

#ifndef SEQFOR_H_
#define SEQFOR_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace qsim {

/**
 * Helper struct for executing for loops in series.
 */
struct SequentialFor {
  explicit SequentialFor(unsigned num_threads) {}

  // SequentialFor does not have any state. So all its methods can be static.

  static uint64_t GetIndex0(uint64_t size, unsigned thread_id) {
    return 0;
  }

  static uint64_t GetIndex1(uint64_t size, unsigned thread_id) {
    return size;
  }

  template <typename Function, typename... Args>
  static void Run(uint64_t size, Function&& func, Args&&... args) {
    for (uint64_t i = 0; i < size; ++i) {
      func(1, 0, i, args...);
    }
  }

  template <typename Function, typename Op, typename... Args>
  static std::vector<typename Op::result_type> RunReduceP(
      uint64_t size, Function&& func, Op&& op, Args&&... args) {
    typename Op::result_type result = 0;

    for (uint64_t i = 0; i < size; ++i) {
      result = op(result, func(1, 0, i, args...));
    }

    return std::vector<typename Op::result_type>(1, result);
  }

  template <typename Function, typename Op, typename... Args>
  static typename Op::result_type RunReduce(uint64_t size, Function&& func,
                                            Op&& op, Args&&... args) {
    return RunReduceP(size, func, std::move(op), args...)[0];
  }
};

}  // namespace qsim

#endif  // SEQFOR_H_
