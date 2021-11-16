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

#ifndef UTIL_CUDA_H_
#define UTIL_CUDA_H_

#include <cuda.h>

#include <cstdlib>

#include "io.h"

namespace qsim {

#define ErrorCheck(code) { ErrorAssert((code), __FILE__, __LINE__); }

inline void ErrorAssert(cudaError_t code, const char* file, unsigned line) {
  if (code != cudaSuccess) {
    IO::errorf("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

template <typename T>
struct Complex {
  __host__ __device__ __forceinline__ Complex() {}

  __host__ __device__ __forceinline__ Complex(const T& re) : re(re), im(0) {}

  __host__ __device__ __forceinline__ Complex(const T& re, const T& im)
      : re(re), im(im) {}

  template <typename U>
  __host__ __device__ __forceinline__ Complex<T>& operator=(
      const Complex<U>& r) {
    re = r.re;
    im = r.im;

    return *this;
  }

  T re;
  T im;
};

template <typename T>
__host__ __device__ __forceinline__ Complex<T> operator+(
    const Complex<T>& l, const Complex<T>& r) {
  return Complex<T>(l.re + r.re, l.im + r.im);
}

template <typename T, typename U>
__host__ __device__ __forceinline__ Complex<T> operator+(
    const Complex<T>& l, const Complex<U>& r) {
  return Complex<T>(l.re + r.re, l.im + r.im);
}

template <typename T>
struct Scalar {
  using type = T;
};

template <typename T>
struct Scalar<Complex<T>> {
  using type = T;
};

template <typename T>
struct Plus {
  template <typename U>
  __device__ __forceinline__ T operator()(const T& v1, const U& v2) const {
    return v1 + v2;
  }
};

template <typename T>
struct Product {
  __device__ __forceinline__ Complex<T> operator()(
      const T& re1, const T& im1, const T& re2, const T& im2) const {
    return Complex<T>(re1 * re2 + im1 * im2, re1 * im2 - im1 * re2);
  }
};

template <typename T>
struct RealProduct {
  __device__ __forceinline__ T operator()(
      const T& re1, const T& im1, const T& re2, const T& im2) const {
    return re1 * re2 + im1 * im2;
  }
};

template <typename FP1, typename Op, unsigned warp_size = 32>
__device__ __forceinline__ FP1 WarpReduce(FP1 val, Op op) {
  for (unsigned i = warp_size / 2; i > 0; i /= 2) {
    val = op(val, __shfl_down_sync(0xffffffff, val, i));
  }

  return val;
}

template <typename FP1, typename Op, unsigned warp_size = 32>
__device__ __forceinline__ Complex<FP1> WarpReduce(Complex<FP1> val, Op op) {
  for (unsigned i = warp_size / 2; i > 0; i /= 2) {
    val.re = op(val.re, __shfl_down_sync(0xffffffff, val.re, i));
    val.im = op(val.im, __shfl_down_sync(0xffffffff, val.im, i));
  }

  return val;
}

}  // namespace qsim

#endif  // UTIL_CUDA_H_
