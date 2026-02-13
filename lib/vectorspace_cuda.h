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

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

#include <memory>
#include <utility>

namespace qsim {

// Use a unique detail namespace to avoid collision with vectorspace.h
namespace cuda_detail {

inline void do_not_free(void*) {}
inline void free(void* ptr) {
  if (ptr != nullptr) {
#ifdef __NVCC__
    ErrorCheck(cudaFree(ptr));
#elif __HIP__
    // Using the qsim ErrorCheck wrapper for HIP
    ErrorCheck(hipFree(ptr));
#endif
  }
}

}  // namespace cuda_detail

// Routines for vector manipulations.
template <typename Impl, typename FP>
class VectorSpaceCUDA {
 public:
  using fp_type = FP;
  // Define Pointer with a clear function pointer type for the deleter
  using Pointer = std::unique_ptr<fp_type, void (*)(void*)>;

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

    static constexpr bool requires_copy_to_host() {
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
    // Use explicit 64-bit calculation for large simulations (>30 qubits) to prevent overflow.
    // Otherwise, keep original behavior to minimize impact.
    uint64_t size;
    if (num_qubits > 30) {
      size = uint64_t{sizeof(fp_type)} * Impl::MinSize(num_qubits);
    } else {
      size = sizeof(fp_type) * Impl::MinSize(num_qubits);
    }

// Ensure we use the correct API based on the compiler
#ifdef __NVCC__
    auto rc = cudaMalloc(&p, size);
#elif __HIP__
    auto rc = hipMalloc(&p, size);
#endif

    if (rc == 0) {  // Success
      return Vector{Pointer{(fp_type*)p, &cuda_detail::free}, num_qubits};
    } else {
      return Null();
    }
  }

  // It is the client's responsibility to make sure that p has at least
  // Impl::MinSize(num_qubits) elements.
  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &cuda_detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &cuda_detail::free}, 0};
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    ErrorCheck(
        cudaMemcpy(dest.get(), src.get(),
                   sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
                   cudaMemcpyDeviceToDevice));

    return true;
  }

  // It is the client's responsibility to make sure that dest has at least
  // Impl::MinSize(src.num_qubits()) elements.
  bool Copy(const Vector& src, fp_type* dest) const {
    ErrorCheck(
        cudaMemcpy(dest, src.get(),
                   sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
                   cudaMemcpyDeviceToHost));

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // Impl::MinSize(dest.num_qubits()) elements.
  bool Copy(const fp_type* src, Vector& dest) const {
    ErrorCheck(
        cudaMemcpy(dest.get(), src,
                   sizeof(fp_type) * Impl::MinSize(dest.num_qubits()),
                   cudaMemcpyHostToDevice));

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // min(size, Impl::MinSize(dest.num_qubits())) elements.
  bool Copy(const fp_type* src, uint64_t size, Vector& dest) const {
    size = std::min(size, Impl::MinSize(dest.num_qubits()));
    ErrorCheck(
        cudaMemcpy(dest.get(), src,
                   sizeof(fp_type) * size,
                   cudaMemcpyHostToDevice));
    return true;
  }

  static void DeviceSync() {
    ErrorCheck(cudaDeviceSynchronize());
  }

 protected:
};

}  // namespace qsim

#endif  // VECTORSPACE_CUDA_H_
