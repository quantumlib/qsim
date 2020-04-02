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
#include <vector>

namespace qsim {

template <uint64_t MIN_SIZE>
struct ParallelForT {
  template <typename Function, typename... Args>
  static void Run(
      unsigned num_threads, uint64_t size, Function&& func, Args&&... args) {
    if (num_threads > 1 && size >= MIN_SIZE) {
      #pragma omp parallel num_threads(num_threads)
      {
        unsigned n = omp_get_num_threads();
        unsigned m = omp_get_thread_num();

        uint64_t i0 = size * m / n;
        uint64_t i1 = size * (m + 1) / n;

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
  static typename Op::result_type RunReduce(unsigned num_threads,
                                            uint64_t size, Function&& func,
                                            Op&& op, Args&&... args) {
    typename Op::result_type result = 0;

    if (num_threads > 1 && size >= MIN_SIZE) {
      std::vector<typename Op::result_type> partial_results(num_threads, 0);

      #pragma omp parallel num_threads(num_threads)
      {
        unsigned n = omp_get_num_threads();
        unsigned m = omp_get_thread_num();

        uint64_t i0 = size * m / n;
        uint64_t i1 = size * (m + 1) / n;

        typename Op::result_type partial_result = 0;

        for (uint64_t i = i0; i < i1; ++i) {
          partial_result = op(partial_result, func(n, m, i, args...));
        }

        partial_results[m] = partial_result;
      }

      for (unsigned i = 0; i < num_threads; ++i) {
        result = op(result, partial_results[i]);
      }
    } else if (num_threads > 0) {
      for (uint64_t i = 0; i < size; ++i) {
        result = op(result, func(1, 0, i, args...));
      }
    }

    return result;
  }
};

using ParallelFor = ParallelForT<1024>;

}  // namespace qsim

#endif  // PARFOR_H_
