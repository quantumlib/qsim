// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COOPERATIVE_FOR_H_
#define COOPERATIVE_FOR_H_

#include <cstdint>

namespace qsim {

// Splits one simulator kernel between the SMT siblings assigned to the same
// state block. ExecuteGateBatchOnBlocks configures the lane before invoking
// the simulator and provides the barrier between consecutive gates.
struct CooperativeFor {
  explicit CooperativeFor(unsigned num_threads) { (void) num_threads; }

  static void Configure(unsigned num_threads, unsigned thread_id) {
    team_size_ = num_threads;
    team_thread_id_ = thread_id;
  }

  template <typename Function, typename... Args>
  static void Run(uint64_t size, Function&& func, Args&&... args) {
    const auto begin = size * team_thread_id_ / team_size_;
    const auto end = size * (team_thread_id_ + 1) / team_size_;
    for (uint64_t i = begin; i < end; ++i) {
      func(team_size_, team_thread_id_, i, args...);
    }
  }

 private:
  inline static thread_local unsigned team_size_ = 1;
  inline static thread_local unsigned team_thread_id_ = 0;
};

}  // namespace qsim

#endif  // COOPERATIVE_FOR_H_
