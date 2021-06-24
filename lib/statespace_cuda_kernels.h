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

#ifndef STATESPACE_CUDA_KERNELS_H_
#define STATESPACE_CUDA_KERNELS_H_

#include <cuda.h>

#include "util_cuda.h"

namespace qsim {

namespace detail {

template <typename FP1, typename FP2,
          typename Op1, typename Op2, typename Op3, unsigned warp_size = 32>
__device__ __forceinline__ FP1 BlockReduce1(
    uint64_t n, Op1 op1, Op2 op2, Op3 op3, const FP2* s1, const FP2* s2) {
  extern __shared__ float shared[];
  FP1* partial1 = (FP1*) shared;

  unsigned tid = threadIdx.x;
  unsigned warp = threadIdx.x / warp_size;
  unsigned lane = threadIdx.x % warp_size;

  uint64_t k0 = 2 * n * blockIdx.x * blockDim.x + 2 * tid - lane;
  uint64_t k1 = k0 + 2 * n * blockDim.x;

  FP1 r;

  r = op1(s1[k0], s1[k0 + warp_size], s2[k0], s2[k0 + warp_size]);
  while ((k0 += 2 * blockDim.x) < k1) {
    r = op2(r, op1(s1[k0], s1[k0 + warp_size], s2[k0], s2[k0 + warp_size]));
  }

  partial1[tid] = r;

  __shared__ FP1 partial2[warp_size];

  if (tid < warp_size) {
    partial2[tid] = 0;
  }

  __syncthreads();

  FP1 val = WarpReduce(partial1[tid], op3);

  if (lane == 0) {
    partial2[warp] = val;
  }

  __syncthreads();

  FP1 result = 0;

  if (tid < warp_size) {
    result = WarpReduce(partial2[tid], op3);
  }

  return result;
}

template <typename FP1, typename FP2,
          typename Op1, typename Op2, typename Op3, unsigned warp_size = 32>
__device__ __forceinline__ FP1 BlockReduce1Masked(
    uint64_t n, uint64_t mask, uint64_t bits, Op1 op1, Op2 op2, Op3 op3,
    const FP2* s1, const FP2* s2) {
  extern __shared__ float shared[];
  FP1* partial1 = (FP1*) shared;

  unsigned tid = threadIdx.x;
  unsigned warp = threadIdx.x / warp_size;
  unsigned lane = threadIdx.x % warp_size;

  uint64_t k0 = 2 * n * blockIdx.x * blockDim.x + 2 * tid - lane;
  uint64_t k1 = k0 + 2 * n * blockDim.x;

  FP1 r = 0;

  if (((k0 + lane) / 2 & mask) == bits) {
    r = op1(s1[k0], s1[k0 + warp_size], s2[k0], s2[k0 + warp_size]);
  }
  while ((k0 += 2 * blockDim.x) < k1) {
    if (((k0 + lane) / 2 & mask) == bits) {
      r = op2(r, op1(s1[k0], s1[k0 + warp_size], s2[k0], s2[k0 + warp_size]));
    }
  }

  partial1[tid] = r;

  __shared__ FP1 partial2[warp_size];

  if (tid < warp_size) {
    partial2[tid] = 0;
  }

  __syncthreads();

  FP1 val = WarpReduce(partial1[tid], op3);

  if (lane == 0) {
    partial2[warp] = val;
  }

  __syncthreads();

  FP1 result = 0;

  if (tid < warp_size) {
    result = WarpReduce(partial2[tid], op3);
  }

  return result;
}

template <typename FP1, typename FP2,
          typename Op2, typename Op3, unsigned warp_size = 32>
__device__ __forceinline__ FP1 BlockReduce2(
    uint64_t n, uint64_t size, Op2 op2, Op3 op3, const FP2* s) {
  extern __shared__ float shared[];
  FP1* partial1 = (FP1*) shared;

  unsigned tid = threadIdx.x;
  uint64_t k0 = n * blockIdx.x * blockDim.x + tid;
  uint64_t k1 = k0 + n * blockDim.x;

  FP1 r = 0;

  if (tid < size) {
    r = s[k0];
    while ((k0 += blockDim.x) < k1) {
      r = op2(r, s[k0]);
    }
  }

  partial1[tid] = r;

  __shared__ FP1 partial2[warp_size];

  if (tid < warp_size) {
    partial2[tid] = 0;
  }

  __syncthreads();

  FP1 val = WarpReduce(partial1[tid], op3);

  if (threadIdx.x % warp_size == 0) {
    partial2[threadIdx.x / warp_size] = val;
  }

  __syncthreads();

  FP1 result = 0;

  if (tid < warp_size) {
    result = WarpReduce(partial2[tid], op3);
  }

  return result;
}

}  // namespace detail

template <typename FP1, typename FP2, typename FP3,
          typename Op1, typename Op2, typename Op3, unsigned warp_size = 32>
__global__ void Reduce1Kernel(uint64_t n, Op1 op1, Op2 op2, Op3 op3,
                              const FP2* s1, const FP2* s2, FP3* result) {
  FP1 sum = detail::BlockReduce1<FP1>(n, op1, op2, op3, s1, s2);

  if (threadIdx.x == 0) {
    result[blockIdx.x] = sum;
  }
}

template <typename FP1, typename FP2, typename FP3,
          typename Op1, typename Op2, typename Op3, unsigned warp_size = 32>
__global__ void Reduce1MaskedKernel(uint64_t n, uint64_t mask, uint64_t bits,
                                    Op1 op1, Op2 op2, Op3 op3,
                                    const FP2* s1, const FP2* s2, FP3* result) {
  FP1 sum =
      detail::BlockReduce1Masked<FP1>(n, mask, bits, op1, op2, op3, s1, s2);

  if (threadIdx.x == 0) {
    result[blockIdx.x] = sum;
  }
}

template <typename FP1, typename FP2, typename FP3,
          typename Op2, typename Op3, unsigned warp_size = 32>
__global__ void Reduce2Kernel(
    uint64_t n, uint64_t size, Op2 op2, Op3 op3, const FP2* s, FP3* result) {
  FP1 sum = detail::BlockReduce2<FP1>(n, size, op2, op3, s);

  if (threadIdx.x == 0) {
    result[blockIdx.x] = sum;
  }
}

template <typename FP, unsigned warp_size = 32>
__global__ void InternalToNormalOrderKernel(FP* state) {
  unsigned lane = threadIdx.x % warp_size;
  unsigned l = 2 * threadIdx.x - lane;
  uint64_t k = 2 * uint64_t{blockIdx.x} * blockDim.x + l;

  extern __shared__ float shared[];
  FP* buf = (FP*) shared;

  buf[l] = state[k];
  buf[l + warp_size] = state[k + warp_size];

  __syncthreads();

  state[k + lane] = buf[l];
  state[k + lane + 1] = buf[l + warp_size];
}

template <typename FP, unsigned warp_size = 32>
__global__ void NormalToInternalOrderKernel(FP* state) {
  unsigned lane = threadIdx.x % warp_size;
  unsigned l = 2 * threadIdx.x - lane;
  uint64_t k = 2 * uint64_t{blockIdx.x} * blockDim.x + l;

  extern __shared__ float shared[];
  FP* buf = (FP*) shared;

  buf[l] = state[k];
  buf[l + warp_size] = state[k + warp_size];

  __syncthreads();

  state[k] = buf[l + lane];
  state[k + warp_size] = buf[l + lane + 1];
}

template <typename FP, unsigned warp_size = 32>
__global__ void SetStateUniformKernel(FP v, uint64_t size, FP* state) {
  unsigned lane = threadIdx.x % warp_size;
  uint64_t k = 2 * (uint64_t{blockIdx.x} * blockDim.x + threadIdx.x) - lane;

  state[k] = lane < size ? v : 0;
  state[k + warp_size] = 0;
}

template <typename FP, unsigned warp_size = 32>
__global__ void AddKernel(const FP* state1, FP* state2) {
  uint64_t k = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  state2[k] += state1[k];
}

template <typename FP, unsigned warp_size = 32>
__global__ void MultiplyKernel(FP a, FP* state) {
  uint64_t k = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  state[k] *= a;
}

template <typename FP, unsigned warp_size = 32>
__global__ void CollapseKernel(uint64_t mask, uint64_t bits, FP r, FP* state) {
  uint64_t k1 = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  uint64_t k2 = 2 * k1 - threadIdx.x % warp_size;

  if ((k1 & mask) == bits) {
    state[k2] *= r;
    state[k2 + warp_size] *= r;
  } else {
    state[k2] = 0;
    state[k2 + warp_size] = 0;
  }
}

template <typename FP, unsigned warp_size = 32>
__global__ void BulkSetAmplKernel(
    uint64_t mask, uint64_t bits, FP re, FP im, bool exclude, FP* state) {
  uint64_t k1 = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  uint64_t k2 = 2 * k1 - threadIdx.x % warp_size;

  bool set = ((k1 & mask) == bits) ^ exclude;

  if (set) {
    state[k2] = re;
    state[k2 + warp_size] = im;
  }
}

template <typename FP1, typename FP2, typename FP3, unsigned warp_size = 32>
__global__ void SampleKernel(unsigned num_blocks,
                             uint64_t n, uint64_t num_samples,
                             const FP1* rs, const FP2* ps, const FP3* state,
                             uint64_t *bitstrings) {
  // Use just one thread. This can be somewhat slow.
  if (threadIdx.x == 0) {
    uint64_t m = 0;
    double csum = 0;

    for (unsigned block_id = 0; block_id < num_blocks; ++block_id) {
      uint64_t km = n * blockDim.x;
      uint64_t k0 = block_id * km;

      for (uint64_t k = 0; k < km; ++k) {
        uint64_t l = 2 * k0 + 64 * (k / 32) + k % 32;
        FP3 re = state[l];
        FP3 im = state[l + warp_size];
        csum += re * re + im * im;
        while (rs[m] < csum && m < num_samples) {
          bitstrings[m++] = k0 + k;
        }
      }
    }
  }
}

template <typename FP, unsigned warp_size = 32>
__global__ void FindMeasuredBitsKernel(
    uint64_t block_id, uint64_t n, double r, const FP* state, uint64_t* res) {
  // Use just one thread. This can be somewhat slow, however, this is
  // more or less consistent with CPU implementations.
  if (threadIdx.x == 0) {
    double csum = 0;
    uint64_t km = n * blockDim.x;
    uint64_t k0 = block_id * km;

    for (uint64_t k = 0; k < km; ++k) {
      uint64_t l = 2 * k0 + 64 * (k / 32) + k % 32;
      FP re = state[l];
      FP im = state[l + warp_size];
      csum += re * re + im * im;
      if (r < csum) {
        *res = k0 + k;
        return;
      }
    }

    *res = k0 + n * blockDim.x - 1;
  }
}

}  // namespace qsim

#endif  // STATESPACE_CUDA_KERNELS_H_
