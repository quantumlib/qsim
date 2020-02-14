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

#ifndef STATESPACE_H_
#define STATESPACE_H_

#ifdef _WIN32
  #include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>

namespace qsim {

namespace detail {

inline void do_not_free(void*) noexcept {}

}  // namespace detail

// Routines for state-vector manipulations.
template <typename ParallelFor, typename FP>
class StateSpace {
 public:
  using fp_type = FP;
  using State = std::unique_ptr<fp_type, decltype(&free)>;

  StateSpace(unsigned num_qubits, unsigned num_threads, uint64_t raw_size)
      : size_(uint64_t{1} << num_qubits), raw_size_(raw_size),
        num_threads_(num_threads) {}

  State CreateState() const {
    auto vector_size = sizeof(fp_type) * raw_size_;
    #ifdef _WIN32
      return State((fp_type*) _aligned_alloc(vector_size, 64), &_aligned_free);
    #else
      void* p = nullptr;
      posix_memalign(&p, 64, vector_size);
      return State((fp_type*) p, &free);
    #endif
  }

  State CreateState(fp_type* p) const {
    return State(p, &detail::do_not_free);
  }

  static State NullState() {
    return State(nullptr, &free);
  }

  uint64_t Size(const State& state) const {
    return size_;
  }

  uint64_t RawSize(const State& state) const {
    return raw_size_;
  }

  static fp_type* RawData(State& state) {
    return state.get();
  }

  static const fp_type* RawData(const State& state) {
    return state.get();
  }

  static bool IsNull(const State& state) {
    return state.get() == nullptr;
  }

  void CopyState(const State& src, State& dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const State& src, State& dest) {
      dest.get()[i] = src.get()[i];
    };

    ParallelFor::Run(num_threads_, raw_size_, f, src, dest);
  }

 protected:
  uint64_t size_;
  uint64_t raw_size_;
  unsigned num_threads_;
};

}  // namespace qsim

#endif  // STATESPACE_H_
