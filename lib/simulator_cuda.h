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

#ifndef SIMULATOR_CUDA_H_
#define SIMULATOR_CUDA_H_

#include "simulator_cuda_kernels.h"

#include <algorithm>
#include <complex>
#include <cstdint>

#include "bits.h"
#include "statespace_cuda.h"
#include "util_cuda.h"

namespace qsim {

/**
 * Quantum circuit simulator with GPU vectorization.
 */
template <typename FP = float>
class SimulatorCUDA final {
 public:
  struct Parameter {
    /**
     * The number of threads per block.
     * Should be 2 to the power of k, where k is in the range [5,8].
     * Note that the number of registers on the multiprocessor can be
     * exceeded if k > 8 (num_threads > 256).
     */
    unsigned num_threads = 256;
  };

  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  explicit SimulatorCUDA(const Parameter& param)
      : param_(param), scratch_(nullptr), scratch_size_(0) {
    ErrorCheck(cudaMalloc(&d_wf, 131072 * sizeof(fp_type)));
    ErrorCheck(cudaMalloc(&d_idx, 992 * sizeof(unsigned)));
    ErrorCheck(cudaMalloc(&d_ms, 7 * sizeof(uint64_t)));
    ErrorCheck(cudaMalloc(&d_xss, 64 * sizeof(uint64_t)));
  }

  ~SimulatorCUDA() {
    ErrorCheck(cudaFree(d_wf));
    ErrorCheck(cudaFree(d_idx));
    ErrorCheck(cudaFree(d_ms));
    ErrorCheck(cudaFree(d_xss));

    if (scratch_ != nullptr) {
      ErrorCheck(cudaFree(scratch_));
    }
  }

  /**
   * Applies a gate using CUDA instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 4) {
        ApplyGate1H(qs, matrix, state);
      } else {
        ApplyGate1L(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 4) {
        ApplyGate2HH(qs, matrix, state);
      } else if (qs[1] > 4) {
        ApplyGate2HL(qs, matrix, state);
      } else {
        ApplyGate2LL(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 4) {
        ApplyGate3HHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        ApplyGate3HHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        ApplyGate3HLL(qs, matrix, state);
      } else {
        ApplyGate3LLL(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 4) {
        ApplyGate4HHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        ApplyGate4HHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        ApplyGate4HHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        ApplyGate4HLLL(qs, matrix, state);
      } else {
        ApplyGate4LLLL(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 4) {
        ApplyGate5HHHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        ApplyGate5HHHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        ApplyGate5HHHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        ApplyGate5HHLLL(qs, matrix, state);
      } else if (qs[4] > 4) {
        ApplyGate5HLLLL(qs, matrix, state);
      } else {
        ApplyGate5LLLLL(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 4) {
        ApplyGate6HHHHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        ApplyGate6HHHHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        ApplyGate6HHHHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        ApplyGate6HHHLLL(qs, matrix, state);
      } else if (qs[4] > 4) {
        ApplyGate6HHLLLL(qs, matrix, state);
      } else {
        ApplyGate6HLLLLL(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using CUDA instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, State& state) const {
    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      if (qs[0] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate1H_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate1H_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 4) {
          ApplyControlledGate1L_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate1L_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 2:
      if (qs[0] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate2HH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2HH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate2HL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2HL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 4) {
          ApplyControlledGate2LL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate2LL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 3:
      if (qs[0] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate3HHH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HHH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate3HHL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HHL_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[2] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate3HLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3HLL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 4) {
          ApplyControlledGate3LLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate3LLL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    case 4:
      if (qs[0] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate4HHHH_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHHH_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[1] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate4HHHL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHHL_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[2] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate4HHLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HHLL_L(qs, cqs, cmask, matrix, state);
        }
      } else if (qs[3] > 4) {
        if (cqs[0] > 4) {
          ApplyControlledGate4HLLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4HLLL_L(qs, cqs, cmask, matrix, state);
        }
      } else {
        if (cqs[0] > 4) {
          ApplyControlledGate4LLLL_H(qs, cqs, cmask, matrix, state);
        } else {
          ApplyControlledGate4LLLL_L(qs, cqs, cmask, matrix, state);
        }
      }
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Computes the expectation value of an operator using CUDA instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      if (qs[0] > 4) {
        return ExpectationValue1H(qs, matrix, state);
      } else {
        return ExpectationValue1L(qs, matrix, state);
      }
      break;
    case 2:
      if (qs[0] > 4) {
        return ExpectationValue2HH(qs, matrix, state);
      } else if (qs[1] > 4) {
        return ExpectationValue2HL(qs, matrix, state);
      } else {
        return ExpectationValue2LL(qs, matrix, state);
      }
      break;
    case 3:
      if (qs[0] > 4) {
        return ExpectationValue3HHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        return ExpectationValue3HHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        return ExpectationValue3HLL(qs, matrix, state);
      } else {
        return ExpectationValue3LLL(qs, matrix, state);
      }
      break;
    case 4:
      if (qs[0] > 4) {
        return ExpectationValue4HHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        return ExpectationValue4HHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        return ExpectationValue4HHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        return ExpectationValue4HLLL(qs, matrix, state);
      } else {
        return ExpectationValue4LLLL(qs, matrix, state);
      }
      break;
    case 5:
      if (qs[0] > 4) {
        return ExpectationValue5HHHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        return ExpectationValue5HHHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        return ExpectationValue5HHHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        return ExpectationValue5HHLLL(qs, matrix, state);
      } else if (qs[4] > 4) {
        return ExpectationValue5HLLLL(qs, matrix, state);
      } else {
        return ExpectationValue5LLLLL(qs, matrix, state);
      }
      break;
    case 6:
      if (qs[0] > 4) {
        return ExpectationValue6HHHHHH(qs, matrix, state);
      } else if (qs[1] > 4) {
        return ExpectationValue6HHHHHL(qs, matrix, state);
      } else if (qs[2] > 4) {
        return ExpectationValue6HHHHLL(qs, matrix, state);
      } else if (qs[3] > 4) {
        return ExpectationValue6HHHLLL(qs, matrix, state);
      } else if (qs[4] > 4) {
        return ExpectationValue6HHLLLL(qs, matrix, state);
      } else {
        return ExpectationValue6HLLLLL(qs, matrix, state);
      }
      break;
    default:
      // Not implemented.
      break;
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 32;
  }

 private:
  void ApplyGate1H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 8 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate1H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate1L(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    unsigned p[32];
    unsigned idx[32];
    fp_type wf[128];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate1L_Kernel<<<blocks, threads>>>(
        d_wf, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate2HH(const std::vector<unsigned>& qs,
                    const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 32 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate2HH_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate2HL(const std::vector<unsigned>& qs,
                    const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate2HL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate2LL(const std::vector<unsigned>& qs,
                    const fp_type* matrix, State& state) const {
    unsigned p[32];
    unsigned idx[96];
    fp_type wf[256];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 256 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate2LL_Kernel<<<blocks, threads>>>(
        d_wf, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate3HHH(const std::vector<unsigned>& qs,
                     const fp_type* matrix, State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate3HHH_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate3HHL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate3HHL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate3HLL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate3HLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate3LLL(const std::vector<unsigned>& qs,
                     const fp_type* matrix, State& state) const {
    unsigned p[32];
    unsigned idx[224];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate3LLL_Kernel<<<blocks, threads>>>(
        d_wf, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate4HHHH(const std::vector<unsigned>& qs,
                      const fp_type* matrix, State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate4HHHH_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate4HHHL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate4HHHL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate4HHLL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate4HHLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate4HLLL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate4HLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate4LLLL(const std::vector<unsigned>& qs,
                      const fp_type* matrix, State& state) const {
    unsigned p[32];
    unsigned idx[480];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 16 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate4LLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5HHHHH(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 10;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5HHHHH_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5HHHHL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[32768];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 32 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 32768 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5HHHHL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5HHHLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[16384];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 32 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 16384 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5HHHLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5HHLLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 32 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5HHLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5HLLLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[4] + 1);
    ms[0] = (uint64_t{1} << qs[4]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (512 * i + 32 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5HLLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate5LLLLL(const std::vector<unsigned>& qs,
                       const fp_type* matrix, State& state) const {
    unsigned p[32];
    unsigned idx[992];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3])
         | (1 << qs[4]);

    for (unsigned i = 0; i < 31; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 32) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (1024 * i + 32 * k + 32 * (m / 32) + (k + m) % 32);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 992 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate5LLLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HHHHHH(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[6];
    uint64_t ms[7];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 6; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[6] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[5] - 1);

    uint64_t xss[64];
    for (unsigned i = 0; i < 64; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 6; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 7 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 64 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 11;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HHHHHH_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HHHHHL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[131072];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 32; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 64 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 131072 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 10;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HHHHHL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HHHHLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[65536];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 64 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 65536 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HHHHLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HHHLLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[32768];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (512 * i + 64 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 32768 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HHHLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HHLLLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[4] + 1);
    ms[0] = (uint64_t{1} << qs[4]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 4] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 4]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[16384];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (1024 * i + 64 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 16384 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HHLLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyGate6HLLLLL(const std::vector<unsigned>& qs,
                        const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[5] + 1);
    ms[0] = (uint64_t{1} << qs[5]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[992];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3])
         | (1 << qs[4]);

    for (unsigned i = 0; i < 31; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 32) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (2048 * i + 64 * k + 32 * (m / 32) + (k + m) % 32);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 992 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyGate6HLLLLL_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate1H_H(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 8 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate1H_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate1H_L(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    fp_type wf[256];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (2 * i + 2 * k + m);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 2 == (p[j] / 2) % 2 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 256 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate1H_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate1L_H(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               State& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[128];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate1L_H_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate1L_L(const std::vector<unsigned>& qs,
                               const std::vector<unsigned>& cqs,
                               uint64_t cmask, const fp_type* matrix,
                               State& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[128];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 2 == (p[j] / 2) % 2 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate1L_L_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2HH_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 32 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2HH_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2HH_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (4 * i + 4 * k + m);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2HH_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2HL_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2HL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2HL_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2HL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2LL_H(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[256];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 256 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2LL_H_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate2LL_L(const std::vector<unsigned>& qs,
                                const std::vector<unsigned>& cqs,
                                uint64_t cmask, const fp_type* matrix,
                                State& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[256];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 4 == (p[j] / 2) % 4 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 256 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate2LL_L_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HHH_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HHH_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HHH_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (8 * i + 8 * k + m);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HHH_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HHL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HHL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HHL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HHL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HLL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HLL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3HLL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3HLL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3LLL_H(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3LLL_H_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate3LLL_L(const std::vector<unsigned>& qs,
                                 const std::vector<unsigned>& cqs,
                                 uint64_t cmask, const fp_type* matrix,
                                 State& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 8 == (p[j] / 2) % 8 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate3LLL_L_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHHH_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 9 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHHH_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHHH_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    fp_type wf[16384];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 16 * k + m);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 16384 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 9 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHHH_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHHL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHHL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHHL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 8 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHHL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHLL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHLL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HHLL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 7 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HHLL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HLLL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HLLL_H_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4HLLL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 6 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4HLLL_L_Kernel<<<blocks, threads>>>(
        d_wf, d_ms, d_xss, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4LLLL_H(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 16 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4LLLL_H_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void ApplyControlledGate4LLLL_L(const std::vector<unsigned>& qs,
                                  const std::vector<unsigned>& cqs,
                                  uint64_t cmask, const fp_type* matrix,
                                  State& state) const {
    unsigned cl = 0;
    uint64_t emaskl = 0;
    uint64_t emaskh = 0;

    for (auto q : cqs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      } else {
        ++cl;
        emaskl |= uint64_t{1} << q;
      }
    }

    uint64_t cmaskh = bits::ExpandBits(cmask >> cl, state.num_qubits(), emaskh);
    uint64_t cmaskl = bits::ExpandBits(cmask & ((1 << cl) - 1), 5, emaskl);

    for (auto q : qs) {
      if (q > 4) {
        emaskh |= uint64_t{1} << q;
      }
    }

    emaskh = ~emaskh ^ 31;

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 16 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          fp_type v = (p[j] / 2) / 16 == (p[j] / 2) % 16 ? 1 : 0;
          wf[32 * l + j] = cmaskl == (j & emaskl) ? matrix[p[j]] : v;
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = cmaskl == (j & emaskl) ? matrix[p[j] + 1] : 0;
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));

    fp_type* rstate = state.get();

    unsigned k = 5 + cqs.size() - cl;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;

    ApplyControlledGate4LLLL_L_Kernel<<<blocks, threads>>>(
        d_wf, state.num_qubits(), cmaskh, emaskh, d_idx, rstate);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  std::complex<double> ExpectationValue1H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 8 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue1H_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue1L(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    unsigned p[32];
    unsigned idx[32];
    fp_type wf[128];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 2; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (4 * i + 2 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (2 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue1L_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue2HH(const std::vector<unsigned>& qs,
                                           const fp_type* matrix,
                                           const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 32 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue2HH_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue2HL(const std::vector<unsigned>& qs,
                                           const fp_type* matrix,
                                           const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (8 * i + 4 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue2HL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue2LL(const std::vector<unsigned>& qs,
                                           const fp_type* matrix,
                                           const State& state) const {
    unsigned p[32];
    unsigned idx[96];
    fp_type wf[256];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 4; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 4 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (4 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 256 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue2LL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue3HHH(const std::vector<unsigned>& qs,
                                            const fp_type* matrix,
                                            const State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 128 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue3HHH_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue3HHL(const std::vector<unsigned>& qs,
                                            const fp_type* matrix,
                                            const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (16 * i + 8 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue3HHL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue3HLL(const std::vector<unsigned>& qs,
                                            const fp_type* matrix,
                                            const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 8 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue3HLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue3LLL(const std::vector<unsigned>& qs,
                                            const fp_type* matrix,
                                            const State& state) const {
    unsigned p[32];
    unsigned idx[224];
    fp_type wf[512];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 8; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 8 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (8 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue3LLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue4HHHH(const std::vector<unsigned>& qs,
                                             const fp_type* matrix,
                                             const State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 512 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue4HHHH_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue4HHHL(const std::vector<unsigned>& qs,
                                             const fp_type* matrix,
                                             const State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (32 * i + 16 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue4HHHL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue4HHLL(const std::vector<unsigned>& qs,
                                             const fp_type* matrix,
                                             const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 16 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue4HHLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue4HLLL(const std::vector<unsigned>& qs,
                                             const fp_type* matrix,
                                             const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 16 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue4HLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue4LLLL(const std::vector<unsigned>& qs,
                                             const fp_type* matrix,
                                             const State& state) const {
    unsigned p[32];
    unsigned idx[480];
    fp_type wf[1024];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 16; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 16 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (16 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 1024 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue4LLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5HHHHH(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 10;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5HHHHH_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5HHHHL(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[32768];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (64 * i + 32 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 32768 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5HHHHL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5HHHLL(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[16384];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 32 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 16384 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5HHHLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5HHLLL(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 32 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5HHLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5HLLLL(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[4] + 1);
    ms[0] = (uint64_t{1} << qs[4]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[4096];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (512 * i + 32 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 4096 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5HLLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue5LLLLL(const std::vector<unsigned>& qs,
                                              const fp_type* matrix,
                                              const State& state) const {
    unsigned p[32];
    unsigned idx[992];
    fp_type wf[2048];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3])
         | (1 << qs[4]);

    for (unsigned i = 0; i < 31; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 32) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned m = 0; m < 32; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (1024 * i + 32 * k + 32 * (m / 32) + (k + m) % 32);
        }

        unsigned l = 2 * (32 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 2048 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 992 * sizeof(unsigned), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue5LLLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HHHHHH(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[6];
    uint64_t ms[7];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 6; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[6] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[5] - 1);

    uint64_t xss[64];
    for (unsigned i = 0; i < 64; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 6; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    ErrorCheck(
        cudaMemcpy(d_wf, matrix, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 7 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 64 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 11;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HHHHHH_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HHHHHL(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[1] + 1);
    ms[0] = (uint64_t{1} << qs[1]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 1] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 1]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[32];
    fp_type wf[131072];

    unsigned qmask = (1 << qs[0]);

    for (unsigned i = 0; i < 1; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 2) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 32; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (128 * i + 64 * k + 2 * (m / 2) + (k + m) % 2);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 131072 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 32 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 32 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 10;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HHHHHL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HHHHLL(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[2] + 1);
    ms[0] = (uint64_t{1} << qs[2]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 2] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 2]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[96];
    fp_type wf[65536];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 4) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 16; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (256 * i + 64 * k + 4 * (m / 4) + (k + m) % 4);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 65536 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 96 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 5 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 9;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HHHHLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HHHLLL(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[3] + 1);
    ms[0] = (uint64_t{1} << qs[3]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 3] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 3]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[224];
    fp_type wf[32768];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]);

    for (unsigned i = 0; i < 7; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 8) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 8; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (512 * i + 64 * k + 8 * (m / 8) + (k + m) % 8);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 32768 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 224 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 8;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HHHLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HHLLLL(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[4] + 1);
    ms[0] = (uint64_t{1} << qs[4]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 4] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 4]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[480];
    fp_type wf[16384];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3]);

    for (unsigned i = 0; i < 15; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 16) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 4; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (1024 * i + 64 * k + 16 * (m / 16) + (k + m) % 16);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 16384 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 480 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 7;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HHLLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  std::complex<double> ExpectationValue6HLLLLL(const std::vector<unsigned>& qs,
                                               const fp_type* matrix,
                                               const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[5] + 1);
    ms[0] = (uint64_t{1} << qs[5]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    unsigned p[32];
    unsigned idx[992];
    fp_type wf[8192];

    unsigned qmask = (1 << qs[0]) | (1 << qs[1]) | (1 << qs[2]) | (1 << qs[3])
         | (1 << qs[4]);

    for (unsigned i = 0; i < 31; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
        idx[32 * i + j] =
            MaskedAdd(j, i + 1, qmask, 32) | (j & (0xffffffff ^ qmask));
      }
    }

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned m = 0; m < 64; ++m) {
        for (unsigned j = 0; j < 32; ++j) {
          unsigned k = bits::CompressBits(j, 5, qmask);
          p[j] = 2 * (2048 * i + 64 * k + 32 * (m / 32) + (k + m) % 32);
        }

        unsigned l = 2 * (64 * i + m);

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j] = matrix[p[j]];
        }

        for (unsigned j = 0; j < 32; ++j) {
          wf[32 * l + j + 32] = matrix[p[j] + 1];
        }
      }
    }

    ErrorCheck(
        cudaMemcpy(d_wf, wf, 8192 * sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_idx, idx, 992 * sizeof(unsigned), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_ms, ms, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(d_xss, xss, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Complex = qsim::Complex<double>;

    unsigned threads = std::min(32 * size, uint64_t{param_.num_threads});
    unsigned blocks = 32 * size / threads;
    unsigned bytes = threads * sizeof(Complex);

    Complex* resd2 = (Complex*) AllocScratch((blocks + 1) * sizeof(Complex));
    Complex* resd1 = resd2 + 1;

    auto op1 = Plus<double>();

    ExpectationValue6HLLLLL_Kernel<<<blocks, threads, bytes>>>(
        d_wf, d_ms, d_xss, d_idx, rstate, op1, resd1);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    Complex result;

    if (blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, resd1, sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
      auto op2 = Plus<Complex>();

      unsigned threads2 = std::min(param_.num_threads, std::max(32U, blocks));
      unsigned dblocks2 = std::max(1U, blocks / threads2);
      unsigned bytes2 = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<1, threads2, bytes2>>>(
          dblocks2, blocks, op2, op1, resd1, resd2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, resd2, sizeof(Complex), cudaMemcpyDeviceToHost));
    }

    return {result.re, result.im};
  }

  static unsigned MaskedAdd(
      unsigned a, unsigned b, unsigned mask, unsigned lsize) {
    unsigned c = bits::CompressBits(a, 5, mask);
    return bits::ExpandBits((c + b) % lsize, 5, mask);
  }

  void* AllocScratch(uint64_t size) const {
    if (size > scratch_size_) {
      if (scratch_ != nullptr) {
        ErrorCheck(cudaFree(scratch_));
      }

      ErrorCheck(cudaMalloc(const_cast<void**>(&scratch_), size));

      const_cast<uint64_t&>(scratch_size_) = size;
    }

    return scratch_;
  }

  Parameter param_;

  fp_type* d_wf;
  unsigned* d_idx;
  uint64_t* d_ms;
  uint64_t* d_xss;

  void* scratch_;
  uint64_t scratch_size_;
};

}  // namespace qsim

#endif  // SIMULATOR_CUDA_H_
