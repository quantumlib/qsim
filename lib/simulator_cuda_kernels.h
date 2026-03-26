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

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>

  #include "util_cuda.h"
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

namespace qsim {

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 64.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                       (G < 6 ? gsize : 32) : (G < 5 ? 8 : 16));

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  idx_type i = (64 * idx_type{blockIdx.x} + threadIdx.x) & 0xffffffffffe0;
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j <= G; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  auto p0 = rstate + 2 * ii + threadIdx.x % 32;

  for (unsigned k = 0; k < gsize; ++k) {
    rs[k] = *(p0 + xss[k]);
    is[k] = *(p0 + xss[k] + 32);
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      __syncthreads();

      for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }

      __syncthreads();
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      *(p0 + xss[k]) = rn;
      *(p0 + xss[k] + 32) = in;
    }
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned esize,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type v[2 * gsize * rows];
  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j <= G; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  auto p0 = rstate + 2 * ii + threadIdx.x;

  for (unsigned k = 0; k < gsize; ++k) {
    rs0[threadIdx.x][k] = *(p0 + xss[k]);
    is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
  }

  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  for (unsigned k = 0; k < esize; ++k) {
    *(p0 + xss[k]) = rs0[threadIdx.x][k];
    *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, unsigned num_mss, idx_type cvalsh,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 64.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                           (G < 6 ? gsize : 32) : (G < 5 ? 8 : 16));

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  idx_type i = (64 * idx_type{blockIdx.x} + threadIdx.x) & 0xffffffffffe0;
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + threadIdx.x % 32;

  for (unsigned k = 0; k < gsize; ++k) {
    rs[k] = *(p0 + xss[k]);
    is[k] = *(p0 + xss[k] + 32);
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      __syncthreads();

      for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }

      __syncthreads();
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      *(p0 + xss[k]) = rn;
      *(p0 + xss[k] + 32) = in;
    }
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateLH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned num_mss, idx_type cvalsh,
    unsigned esize, fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + threadIdx.x;

  for (unsigned k = 0; k < gsize; ++k) {
    rs0[threadIdx.x][k] = *(p0 + xss[k]);
    is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
  }

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  for (unsigned k = 0; k < esize; ++k) {
    *(p0 + xss[k]) = rs0[threadIdx.x][k];
    *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, const idx_type* __restrict__ cis,
    unsigned num_mss, idx_type cvalsh, unsigned esize, unsigned rwthreads,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + cis[threadIdx.x];

  if (threadIdx.x < rwthreads) {
    for (unsigned k = 0; k < gsize; ++k) {
      rs0[threadIdx.x][k] = *(p0 + xss[k]);
      is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
    }
  }

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  if (threadIdx.x < rwthreads) {
    for (unsigned k = 0; k < esize; ++k) {
      *(p0 + xss[k]) = rs0[threadIdx.x][k];
      *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
    }
  }
}

template <unsigned G, typename fp_type, typename idx_type, typename Op,
          typename cfp_type>
__global__ void ExpectationValueH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, unsigned num_iterations_per_block,
    const fp_type* __restrict__ rstate, Op op, cfp_type* __restrict__ result) {
  // blockDim.x must be equal to 64.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 5 ? gsize : (sizeof(fp_type) == 4 ? (G < 6 ? 4 : 8) : 8);

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  double re = 0;
  double im = 0;

  for (unsigned iter = 0; iter < num_iterations_per_block; ++iter) {
    idx_type b = num_iterations_per_block * idx_type{blockIdx.x} + iter;

    idx_type i = (64 * b + threadIdx.x) & 0xffffffffffe0;
    idx_type ii = i & mss[0];
    for (unsigned j = 1; j <= G; ++j) {
      i *= 2;
      ii |= i & mss[j];
    }

    auto p0 = rstate + 2 * ii + threadIdx.x % 32;

    for (unsigned k = 0; k < gsize; ++k) {
      rs[k] = *(p0 + xss[k]);
      is[k] = *(p0 + xss[k] + 32);
    }

    for (unsigned s = 0; s < gsize / rows; ++s) {
      if (s > 0 || iter > 0) {
        __syncthreads();

        for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
          v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
        }

        __syncthreads();
      }

      unsigned j = 0;

      for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
        fp_type rn = 0;
        fp_type in = 0;

        for (unsigned l = 0; l < gsize; ++l) {
          fp_type rm = v[j++];
          fp_type im = v[j++];
          rn += rs[l] * rm;
          rn -= is[l] * im;
          in += rs[l] * im;
          in += is[l] * rm;
        }

        re += rs[k] * rn;
        re += is[k] * in;
        im += rs[k] * in;
        im -= is[k] * rn;
      }
    }
  }

  __shared__ cfp_type partial1[64];
  __shared__ cfp_type partial2[2];

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (threadIdx.x % 32 == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    result[blockIdx.x].re = partial2[0].re + partial2[1].re;
    result[blockIdx.x].im = partial2[0].im + partial2[1].im;
  }
}

template <unsigned G, typename fp_type, typename idx_type,
          typename Op, typename cfp_type>
__global__ void ExpectationValueL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned num_iterations_per_block,
    const fp_type* __restrict__ rstate, Op op, cfp_type* __restrict__ result) {
  // blockDim.x must be equal to 32.

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows = G < 5 ? gsize : (sizeof(fp_type) == 4 ?
                                             (G < 6 ? 4 : 2) : (G < 6 ? 2 : 1));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  double re = 0;
  double im = 0;

  for (idx_type iter = 0; iter < num_iterations_per_block; ++iter) {
    idx_type i = 32 * (num_iterations_per_block * idx_type{blockIdx.x} + iter);
    idx_type ii = i & mss[0];
    for (unsigned j = 1; j <= G; ++j) {
      i *= 2;
      ii |= i & mss[j];
    }

    auto p0 = rstate + 2 * ii + threadIdx.x;

    for (unsigned k = 0; k < gsize; ++k) {
      rs0[threadIdx.x][k] = *(p0 + xss[k]);
      is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
    }

    for (unsigned k = 0; k < gsize; ++k) {
      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs[k] = rs0[m][n];
      is[k] = is0[m][n];
    }

    for (unsigned s = 0; s < gsize / rows; ++s) {
      if (s > 0 || iter > 0) {
        for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
          v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
        }
      }

      unsigned j = 0;

      for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
        fp_type rn = 0;
        fp_type in = 0;

        for (unsigned l = 0; l < gsize; ++l) {
          fp_type rm = v[j++];
          fp_type im = v[j++];
          rn += rs[l] * rm;
          rn -= is[l] * im;
          in += rs[l] * im;
          in += is[l] * rm;
        }

        re += rs[k] * rn;
        re += is[k] * in;
        im += rs[k] * in;
        im -= is[k] * rn;
      }
    }
  }

  __shared__ cfp_type partial[32];

  partial[threadIdx.x].re = re;
  partial[threadIdx.x].im = im;

  auto val = WarpReduce(partial[threadIdx.x], op);

  if (threadIdx.x == 0) {
    result[blockIdx.x].re = val.re;
    result[blockIdx.x].im = val.im;
  }
}

}  // namespace qsim

#endif  // SIMULATOR_CUDA_KERNELS_H_
