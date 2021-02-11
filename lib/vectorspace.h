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

#ifndef VECTORSPACE_H_
#define VECTORSPACE_H_

#ifdef _WIN32
  #include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

namespace qsim {

namespace detail {

inline void do_not_free(void*) noexcept {}

inline void free(void* ptr) noexcept {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename For, typename FP>
class VectorSpace {
 public:
  using fp_type = FP;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&free)>;

 public:
  class Vector {
   public:
    Vector() = delete;

    Vector(Pointer&& ptr, unsigned num_qubits)
        : ptr_(std::move(ptr)), num_qubits_(num_qubits) {}

    fp_type* get() {
      return ptr_.get();
    }

    const fp_type* get() const {
      return ptr_.get();
    }

    fp_type* release() {
      num_qubits_ = 0;
      return ptr_.release();
    }

    unsigned num_qubits() const {
      return num_qubits_;
    }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
  };

  template <typename... ForArgs>
  VectorSpace(ForArgs&&... args) : for_(args...) {}

  static Vector Create(unsigned num_qubits) {
    auto size = sizeof(fp_type) * Impl::MinSize(num_qubits);
    #ifdef _WIN32
      Pointer ptr{(fp_type*) _aligned_malloc(size, 64), &detail::free};
      return Vector{std::move(ptr), ptr.get() != nullptr ? num_qubits : 0};
    #else
      void* p = nullptr;
      if (posix_memalign(&p, 64, size) == 0) {
        return Vector{Pointer{(fp_type*) p, &detail::free}, num_qubits};
      } else {
        return Null();
      }
    #endif
  }

  // It is the client's responsibility to make sure that p has at least
  // 2 * 2^num_qubits elements.
  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &detail::free}, 0};
  }

  static bool IsNull(const Vector& vec) {
    return vec.get() == nullptr;
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    for_.Run(Impl::MinSize(src.num_qubits()), f, src.get(), dest.get());

    return true;
  }

 protected:
  For for_;
};

}  // namespace qsim

#endif  // VECTORSPACE_H_
