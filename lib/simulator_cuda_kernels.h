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

#ifndef SIMULATOR_CUDA_KERNELS_H_
#define SIMULATOR_CUDA_KERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#include "util_cuda.h"

namespace qsim {

template <typename Integer>
__device__ __forceinline__ Integer ExpandBits(
    Integer bits, unsigned n, Integer mask) {
  Integer ebits = 0;
  unsigned k = 0;

  for (unsigned i = 0; i < n; ++i) {
    if ((mask >> i) & 1) {
      ebits |= ((bits >> k) & 1) << i;
      ++k;
    }
  }

  return ebits;
}

template <typename fp_type>
__global__ void ApplyGate1H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate1L_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[2 * l] = *(p0);
    is[2 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate2HH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate2HL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate2LL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[4 * l] = *(p0);
    is[4 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate3HHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate3HHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate3HLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate3LLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[8 * l] = *(p0);
    is[8 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate4HHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate4HHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate4HHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate4HLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate4LLLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[16 * l] = *(p0);
    is[16 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5HHHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 32; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 32; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5HHHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5HHHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5HHLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5HLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[16 * l] = *(p0 + xss[l]);
    is[16 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate5LLLLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[32 * l] = *(p0);
    is[32 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 32; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[32 * l + j] = __shfl_sync(0xffffffff, rs[32 * l], idx[k]);
      is[32 * l + j] = __shfl_sync(0xffffffff, is[32 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HHHHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5])
      | (2048 * i & ms[6]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 64; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 64; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HHHHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 32; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 32; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HHHHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HHHLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HHLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[16 * l] = *(p0 + xss[l]);
    is[16 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyGate6HLLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[32 * l] = *(p0 + xss[l]);
    is[32 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 32; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[32 * l + j] = __shfl_sync(0xffffffff, rs[32 * l], idx[k]);
      is[32 * l + j] = __shfl_sync(0xffffffff, is[32 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate1H_H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate1H_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate1L_H_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[2 * l] = *(p0);
    is[2 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate1L_L_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[2 * l] = *(p0);
    is[2 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2HH_H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2HH_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2HL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2HL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2LL_H_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[4 * l] = *(p0);
    is[4 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate2LL_L_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[4 * l] = *(p0);
    is[4 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HHH_H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HHH_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HHL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HHL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HLL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3HLL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3LLL_H_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[8 * l] = *(p0);
    is[8 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate3LLL_L_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[8 * l] = *(p0);
    is[8 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHHH_H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = 0;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHHH_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHHL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHHL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHLL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HHLL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HLLL_H_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4HLLL_L_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0 + xss[l]) = rn;
    *(p0 + xss[l] + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4LLLL_H_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[16 * l] = *(p0);
    is[16 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type>
__global__ void ApplyControlledGate4LLLL_L_Kernel(
    const fp_type* __restrict__ w, unsigned num_qubits,
    uint64_t cmaskh, uint64_t emaskh, const unsigned* __restrict__ idx,
    fp_type* rstate) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = ExpandBits(i, num_qubits, emaskh) | cmaskh;
  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[16 * l] = *(p0);
    is[16 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    *(p0) = rn;
    *(p0 + 32) = in;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue1H_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue1L_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[2], is[2];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[2 * l] = *(p0);
    is[2 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 2; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue2HH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue2HL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 2 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue2LL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[4], is[4];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[4 * l] = *(p0);
    is[4 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 4; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue3HHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue3HHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 2 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue3HLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 4 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue3LLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[8], is[8];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[8 * l] = *(p0);
    is[8 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 8; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue4HHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue4HHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 2 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue4HHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 4 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue4HLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 8 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue4LLLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[16], is[16];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[16 * l] = *(p0);
    is[16 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 16; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5HHHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 32; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 32; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5HHHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 2 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5HHHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 4 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5HHLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 8 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5HLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[16 * l] = *(p0 + xss[l]);
    is[16 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 16 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue5LLLLL_Kernel(
    const fp_type* __restrict__ w, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[32], is[32];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  auto p0 = rstate + 64 * i + lane;

  for (unsigned l = 0; l < 1; ++l) {
    rs[32 * l] = *(p0);
    is[32 * l] = *(p0 + 32);

    for (unsigned j = 1; j < 32; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[32 * l + j] = __shfl_sync(0xffffffff, rs[32 * l], idx[k]);
      is[32 * l + j] = __shfl_sync(0xffffffff, is[32 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 1; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 32; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HHHHHH_Kernel(
    const fp_type* __restrict__ v, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5])
      | (2048 * i & ms[6]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 64; ++l) {
    rs[l] = *(p0 + xss[l]);
    is[l] = *(p0 + xss[l] + 32);
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = 0;

  for (unsigned l = 0; l < 64; ++l) {
    rn = rs[0] * v[j] - is[0] * v[j + 1];
    in = rs[0] * v[j + 1] + is[0] * v[j];

    j += 2;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * v[j] - is[n] * v[j + 1];
      in += rs[n] * v[j + 1] + is[n] * v[j];

      j += 2;
    }

    re += rs[l] * rn + is[l] * in;
    im += rs[l] * in - is[l] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HHHHHL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]) | (1024 * i & ms[5]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 32; ++l) {
    rs[2 * l] = *(p0 + xss[l]);
    is[2 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 2; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[2 * l + j] = __shfl_sync(0xffffffff, rs[2 * l], idx[k]);
      is[2 * l + j] = __shfl_sync(0xffffffff, is[2 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 32; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 2 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HHHHLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]) | (512 * i & ms[4]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 16; ++l) {
    rs[4 * l] = *(p0 + xss[l]);
    is[4 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 4; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[4 * l + j] = __shfl_sync(0xffffffff, rs[4 * l], idx[k]);
      is[4 * l + j] = __shfl_sync(0xffffffff, is[4 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 16; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 4 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HHHLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2])
      | (256 * i & ms[3]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 8; ++l) {
    rs[8 * l] = *(p0 + xss[l]);
    is[8 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 8; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[8 * l + j] = __shfl_sync(0xffffffff, rs[8 * l], idx[k]);
      is[8 * l + j] = __shfl_sync(0xffffffff, is[8 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 8; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 8 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HHLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]) | (128 * i & ms[2]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 4; ++l) {
    rs[16 * l] = *(p0 + xss[l]);
    is[16 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 16; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[16 * l + j] = __shfl_sync(0xffffffff, rs[16 * l], idx[k]);
      is[16 * l + j] = __shfl_sync(0xffffffff, is[16 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 4; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 16 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

template <typename fp_type, typename Op, typename FP>
__global__ void ExpectationValue6HLLLLL_Kernel(
    const fp_type* __restrict__ w, const uint64_t* __restrict__ ms,
    const uint64_t* __restrict__ xss, const unsigned* __restrict__ idx,
    const fp_type* rstate,
    Op op, FP* result) {
  fp_type rn, in;
  fp_type rs[64], is[64];

  unsigned lane = threadIdx.x % 32;
  uint64_t i = (uint64_t{blockDim.x} * blockIdx.x + threadIdx.x) / 32;

  uint64_t k = (32 * i & ms[0]) | (64 * i & ms[1]);

  auto p0 = rstate + 2 * k + lane;

  for (unsigned l = 0; l < 2; ++l) {
    rs[32 * l] = *(p0 + xss[l]);
    is[32 * l] = *(p0 + xss[l] + 32);

    for (unsigned j = 1; j < 32; ++j) {
      unsigned k = 32 * (j - 1) + lane;
      rs[32 * l + j] = __shfl_sync(0xffffffff, rs[32 * l], idx[k]);
      is[32 * l + j] = __shfl_sync(0xffffffff, is[32 * l], idx[k]);
    }
  }

  fp_type re = 0;
  fp_type im = 0;

  unsigned j = lane;

  for (unsigned l = 0; l < 2; ++l) {
    rn = rs[0] * w[j] - is[0] * w[j + 32];
    in = rs[0] * w[j + 32] + is[0] * w[j];

    j += 64;

    for (unsigned n = 1; n < 64; ++n) {
      rn += rs[n] * w[j] - is[n] * w[j + 32];
      in += rs[n] * w[j + 32] + is[n] * w[j];

      j += 64;
    }

    unsigned m = 32 * l;

    re += rs[m] * rn + is[m] * in;
    im += rs[m] * in - is[m] * rn;
  }

  extern __shared__ float shared[];
  FP* partial1 = (FP*) shared;

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  __shared__ FP partial2[32];

  if (threadIdx.x < 32) {
    partial2[threadIdx.x] = 0;
  }

  __syncthreads();

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (lane == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  FP r = 0;

  if (threadIdx.x < 32) {
    r = WarpReduce(partial2[lane], op);
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = r;
  }
};

}  // namespace qsim

#endif  // SIMULATOR_CUDA_KERNELS_H_
