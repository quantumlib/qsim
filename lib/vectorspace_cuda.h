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

#ifndef VECTORSPACE_CUDA_H_
#define VECTORSPACE_CUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <utility>

namespace qsim {

namespace detail {

inline void do_not_free(void*) {}

inline void free(void* ptr) {
  cudaFree(ptr);
}

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename FP>
class VectorSpaceCUDA {
 public:
  using fp_type = FP;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;

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

    bool requires_copy_to_host() const {
      return true;
    }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
  };

  template <typename... Args>
  VectorSpaceCUDA(Args&&... args) {}

  static Vector Create(unsigned num_qubits) {
    fp_type* p;
    auto size = sizeof(fp_type) * Impl::MinSize(num_qubits);
    auto rc = cudaMalloc(&p, size);

    if (rc == cudaSuccess) {
      return Vector{Pointer{(fp_type*) p, &detail::free}, num_qubits};
    } else {
      return Null();
    }
  }

  // It is the client's responsibility to make sure that p has at least
  // 2 * 2^num_qubits elements.
  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &detail::free}, 0};
  }

  static bool IsNull(const Vector& vector) {
    return vector.get() == nullptr;
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    cudaMemcpy(dest.get(), src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToDevice);

    return true;
  }

  // It is the client's responsibility to make sure that dest has at least
  // 2 * 2^src.num_qubits() elements.
  bool Copy(const Vector& src, fp_type* dest) const {
    cudaMemcpy(dest, src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToHost);

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // 2 * 2^dest.num_qubits() elements.
  bool Copy(const fp_type* src, Vector& dest) const {
    cudaMemcpy(dest.get(), src,
               sizeof(fp_type) * Impl::MinSize(dest.num_qubits()),
               cudaMemcpyHostToDevice);

    return true;
  }

 protected:
};

}  // namespace qsim

#endif  // VECTORSPACE_CUDA_H_
