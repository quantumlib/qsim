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
#include <vector>

namespace qsim {

namespace detail {

inline void do_not_free(void*) noexcept {}

}  // namespace detail

// Routines for vec-vector manipulations.
template <typename For, typename FP>
class VectorSpace {
 public:
  using fp_type = FP;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&free)>;

 public:
  class Vector {
   public:
    Vector(Pointer&& ptr, uint64_t size) : ptr_(std::move(ptr)), size_(size) {}

    fp_type* get() {
      return ptr_.get();
    }

    const fp_type* get() const {
      return ptr_.get();
    }

    uint64_t size() const {
      return size_;
    }

   private:
    Pointer ptr_;
    uint64_t size_;
  };

  template <typename... ForArgs>
  VectorSpace(uint64_t raw_size, ForArgs&&... args)
      : for_(args...), raw_size_(raw_size) {}

  Vector CreateVector() const {
    auto size = sizeof(fp_type) * raw_size_;
    #ifdef _WIN32
      Pointer ptr{(fp_type*) _aligned_malloc(size, 64), &_aligned_free};
      uint64_t true_size = ptr.get() != nullptr ? raw_size_ : 0;
      return Vector{std::move(ptr), true_size};
    #else
      void* p = nullptr;
      if (posix_memalign(&p, 64, size) == 0) {
        return Vector{Pointer{(fp_type*) p, &free}, raw_size_};
      } else {
        return Vector{Pointer{nullptr, &free}, 0};
      }
    #endif
  }

  static Vector CreateVector(fp_type* p, uint64_t size) {
    return Vector{Pointer{p, &detail::do_not_free}, size};
  }

  static Vector NullVector() {
    return Vector{Pointer{nullptr, &free}, 0};
  }

  uint64_t RawSize() const {
    return raw_size_;
  }

  static fp_type* RawData(Vector& vec) {
    return vec.get();
  }

  static const fp_type* RawData(const Vector& vec) {
    return vec.get();
  }

  static bool IsNull(const Vector& vec) {
    return vec.get() == nullptr;
  }

  bool Swap(Vector& vec1, Vector& vec2) const {
    if (vec1.size() != raw_size_ || vec2.size() != raw_size_) {
      return false;
    } else {
      std::swap(vec1, vec2);
      return true;
    }
  }

  bool CopyVector(const Vector& src, Vector& dest) const {
    if (src.size() != raw_size_ || dest.size() != raw_size_) {
      return false;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    for_.Run(raw_size_, f, src.get(), dest.get());

    return true;
  }

 protected:
  For for_;
  uint64_t raw_size_;
};

}  // namespace qsim

#endif  // VECTORSPACE_H_
