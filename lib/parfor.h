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

#ifndef PARFOR_H_
#define PARFOR_H_

#include <omp.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace qsim {

/**
 * Helper struct for executing for-loops in parallel across multiple threads.
 */
template <uint64_t MIN_SIZE>
struct ParallelForT {
  explicit ParallelForT(unsigned num_threads) : num_threads(num_threads) {}

  // GetIndex0 and GetIndex1 are useful when we need to know how work was
  // divided between threads, for instance, for reusing partial sums obtained
  // by RunReduceP.
  uint64_t GetIndex0(uint64_t size, unsigned thread_id) const {
    return size >= MIN_SIZE ? size * thread_id / num_threads : 0;
  }

  uint64_t GetIndex1(uint64_t size, unsigned thread_id) const {
    return size >= MIN_SIZE ? size * (thread_id + 1) / num_threads : size;
  }

  template <typename Function, typename... Args>
  void Run(uint64_t size, Function&& func, Args&&... args) const {
    if (num_threads > 1 && size >= MIN_SIZE) {
      #pragma omp parallel num_threads(num_threads)
      {
        unsigned n = omp_get_num_threads();
        unsigned m = omp_get_thread_num();

        uint64_t i0 = GetIndex0(size, m);
        uint64_t i1 = GetIndex1(size, m);

        for (uint64_t i = i0; i < i1; ++i) {
          func(n, m, i, args...);
        }
      }
    } else {
      for (uint64_t i = 0; i < size; ++i) {
        func(1, 0, i, args...);
      }
    }
  }

  template <typename Function, typename Op, typename... Args>
  std::vector<typename Op::result_type> RunReduceP(
      uint64_t size, Function&& func, Op&& op, Args&&... args) const {
    std::vector<typename Op::result_type> partial_results;

    if (num_threads > 1 && size >= MIN_SIZE) {
      partial_results.resize(num_threads, 0);

      #pragma omp parallel num_threads(num_threads)
      {
        unsigned n = omp_get_num_threads();
        unsigned m = omp_get_thread_num();

        uint64_t i0 = GetIndex0(size, m);
        uint64_t i1 = GetIndex1(size, m);

        typename Op::result_type partial_result = 0;

        for (uint64_t i = i0; i < i1; ++i) {
          partial_result = op(partial_result, func(n, m, i, args...));
        }

        partial_results[m] = partial_result;
      }
    } else if (num_threads > 0) {
      typename Op::result_type result = 0;
      for (uint64_t i = 0; i < size; ++i) {
        result = op(result, func(1, 0, i, args...));
      }

      partial_results.resize(1, result);
    }

    return partial_results;
  }

  template <typename Function, typename Op, typename... Args>
  typename Op::result_type RunReduce(uint64_t size, Function&& func,
                                     Op&& op, Args&&... args) const {
    auto partial_results = RunReduceP(size, func, std::move(op), args...);

    typename Op::result_type result = 0;

    for (auto partial_result : partial_results) {
      result = op(result, partial_result);
    }

    return result;
  }

  unsigned num_threads;
};

using ParallelFor = ParallelForT<1024>;

}  // namespace qsim

#endif  // PARFOR_H_
