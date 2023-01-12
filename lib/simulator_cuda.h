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
#include <cstring>
#include <vector>

#include "bits.h"
#include "statespace_cuda.h"

namespace qsim {

/**
 * Quantum circuit simulator with GPU vectorization.
 */
template <typename FP = float>
class SimulatorCUDA final {
 private:
  using idx_type = uint64_t;
  using Complex = qsim::Complex<double>;

  // The maximum buffer size for indices and gate matrices.
  // The maximum gate matrix size (for 6-qubit gates) is
  // 2 * 2^6 * 2^6 * sizeof(FP) = 8192 * sizeof(FP). The maximum index size is
  // 128 * sizeof(idx_type) + 96 * sizeof(unsigned).
  static constexpr unsigned max_buf_size = 8192 * sizeof(FP)
      + 128 * sizeof(idx_type) + 96 * sizeof(unsigned);

 public:
  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  SimulatorCUDA() : scratch_(nullptr), scratch_size_(0) {
    ErrorCheck(cudaMalloc(&d_ws, max_buf_size));
  }

  ~SimulatorCUDA() {
    ErrorCheck(cudaFree(d_ws));

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

    if (qs.size() == 0) {
      ApplyGateH<0>(qs, matrix, state);
    } else if (qs[0] > 4) {
      switch (qs.size()) {
      case 1:
        ApplyGateH<1>(qs, matrix, state);
        break;
      case 2:
        ApplyGateH<2>(qs, matrix, state);
        break;
      case 3:
        ApplyGateH<3>(qs, matrix, state);
        break;
      case 4:
        ApplyGateH<4>(qs, matrix, state);
        break;
      case 5:
        ApplyGateH<5>(qs, matrix, state);
        break;
      case 6:
        ApplyGateH<6>(qs, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    } else {
      switch (qs.size()) {
      case 1:
        ApplyGateL<1>(qs, matrix, state);
        break;
      case 2:
        ApplyGateL<2>(qs, matrix, state);
        break;
      case 3:
        ApplyGateL<3>(qs, matrix, state);
        break;
      case 4:
        ApplyGateL<4>(qs, matrix, state);
        break;
      case 5:
        ApplyGateL<5>(qs, matrix, state);
        break;
      case 6:
        ApplyGateL<6>(qs, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    }
  }

  /**
   * Applies a controlled gate using CUDA instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    if (cqs[0] < 5) {
      switch (qs.size()) {
      case 0:
        ApplyControlledGateL<0>(qs, cqs, cvals, matrix, state);
        break;
      case 1:
        ApplyControlledGateL<1>(qs, cqs, cvals, matrix, state);
        break;
      case 2:
        ApplyControlledGateL<2>(qs, cqs, cvals, matrix, state);
        break;
      case 3:
        ApplyControlledGateL<3>(qs, cqs, cvals, matrix, state);
        break;
      case 4:
        ApplyControlledGateL<4>(qs, cqs, cvals, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    } else {
      if (qs.size() == 0) {
        ApplyControlledGateHH<0>(qs, cqs, cvals, matrix, state);
      } else if (qs[0] > 4) {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      } else {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateLH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateLH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateLH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateLH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      }
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

    if (qs[0] > 4) {
      switch (qs.size()) {
      case 1:
        return ExpectationValueH<1>(qs, matrix, state);
      case 2:
        return ExpectationValueH<2>(qs, matrix, state);
      case 3:
        return ExpectationValueH<3>(qs, matrix, state);
      case 4:
        return ExpectationValueH<4>(qs, matrix, state);
      case 5:
        return ExpectationValueH<5>(qs, matrix, state);
      case 6:
        return ExpectationValueH<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
    } else {
      switch (qs.size()) {
      case 1:
        return ExpectationValueL<1>(qs, matrix, state);
      case 2:
        return ExpectationValueL<2>(qs, matrix, state);
      case 3:
        return ExpectationValueL<3>(qs, matrix, state);
      case 4:
        return ExpectationValueL<4>(qs, matrix, state);
      case 5:
        return ExpectationValueL<5>(qs, matrix, state);
      case 6:
        return ExpectationValueL<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
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
  // The following indices are used in kernels.
  // xss - indices to access the state vector entries in global memory.
  // ms  - masks to access the state vector entries in global memory.
  // tis - indices to access the state vector entries in shared memory
  //       in the presence of low gate qubits.
  // qis - indices to access the state vector entries in shared memory
  //       in the presence of low gate qubits.
  // cis - additional indices to access the state vector entries in global
  //       memory in the presence of low control qubits.

  template <unsigned G>
  struct IndicesH {
    static constexpr unsigned gsize = 1 << G;
    static constexpr unsigned matrix_size = 2 * gsize * gsize * sizeof(fp_type);
    static constexpr unsigned xss_size = 32 * sizeof(idx_type) * (1 + (G == 6));
    static constexpr unsigned ms_size = 32 * sizeof(idx_type);
    static constexpr unsigned xss_offs = matrix_size;
    static constexpr unsigned ms_offs = xss_offs + xss_size;
    static constexpr unsigned buf_size = ms_offs + ms_size;

    IndicesH(char* p)
        : xss((idx_type*) (p + xss_offs)), ms((idx_type*) (p + ms_offs)) {}

    idx_type* xss;
    idx_type* ms;
  };

  template <unsigned G>
  struct IndicesL : public IndicesH<G> {
    using Base = IndicesH<G>;
    static constexpr unsigned qis_size = 32 * sizeof(unsigned) * (1 + (G == 6));
    static constexpr unsigned tis_size = 32 * sizeof(unsigned);
    static constexpr unsigned qis_offs = Base::buf_size;
    static constexpr unsigned tis_offs = qis_offs + qis_size;
    static constexpr unsigned buf_size = tis_offs + tis_size;

    IndicesL(char* p)
        : Base(p), qis((unsigned*) (p + qis_offs)),
          tis((unsigned*) (p + tis_offs)) {}

    unsigned* qis;
    unsigned* tis;
  };

  template <unsigned G>
  struct IndicesLC : public IndicesL<G> {
    using Base = IndicesL<G>;
    static constexpr unsigned cis_size = 32 * sizeof(idx_type);
    static constexpr unsigned cis_offs = Base::buf_size;
    static constexpr unsigned buf_size = cis_offs + cis_size;

    IndicesLC(char* p) : Base(p), cis((idx_type*) (p + cis_offs)) {}

    idx_type* cis;
  };

  struct DataC {
    idx_type cvalsh;
    unsigned num_aqs;
    unsigned num_effective_qs;
    unsigned remaining_low_cqs;
  };

  template <unsigned G>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    IndicesH<G> d_i(d_ws);

    ApplyGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, state.get());
  }

  template <unsigned G>
  void ApplyGateL(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesL<G> d_i(d_ws);

    ApplyGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        1 << num_effective_qs, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, idx_type cvals,
                             const fp_type* matrix, State& state) const {
    unsigned aqs[64];
    idx_type cmaskh = 0;
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);

    unsigned num_aqs = GetHighQubits(qs, 0, cqs, 0, 0, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, h_i.ms);
    GetXss(num_qubits, qs, qs.size(), h_i.xss);

    idx_type cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    IndicesH<G> d_i(d_ws);

    ApplyControlledGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_aqs + 1, cvalsh, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateLH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto d = GetIndicesLC(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesL<G> d_i(d_ws);

    ApplyControlledGateLH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesLC<G> h_i(h_ws);
    auto d = GetIndicesLCL(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesLC<G> d_i(d_ws);

    ApplyControlledGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis, d_i.cis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs,
        1 << (5 - d.remaining_low_cqs), state.get());
  }

  template <unsigned G>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 14 ? n - 14 : 0, 4U);
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, (size / 2) >> s);
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;

    IndicesH<G> d_i(d_ws);

    ExpectationValueH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_iterations_per_block,
        state.get(), Plus<double>(), d_res1);

    double mul = size == 1 ? 0.5 : 1.0;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned G>
  std::complex<double> ExpectationValueL(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice));

    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 13 ? n - 13 : 0, 5U);
    unsigned threads = 32;
    unsigned blocks = size >> s;
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;

    IndicesL<G> d_i(d_ws);

    ExpectationValueL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        num_iterations_per_block, state.get(), Plus<double>(), d_res1);

    double mul = double(1 << (5 + num_effective_qs - G)) / 32;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned m>
  std::complex<double> ExpectationValueReduceFinal(
      unsigned blocks, double mul,
      const Complex* d_res1, Complex* d_res2) const {
    Complex res2[m];

    if (blocks <= 16) {
      ErrorCheck(cudaMemcpy(res2, d_res1, blocks * sizeof(Complex),
                            cudaMemcpyDeviceToHost));
    } else {
      unsigned threads2 = std::min(1024U, blocks);
      unsigned blocks2 = std::min(m, blocks / threads2);

      unsigned dblocks = std::max(1U, blocks / (blocks2 * threads2));
      unsigned bytes = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<blocks2, threads2, bytes>>>(
          dblocks, blocks, Plus<Complex>(), Plus<double>(), d_res1, d_res2);

      ErrorCheck(cudaMemcpy(res2, d_res2, blocks2 * sizeof(Complex),
                            cudaMemcpyDeviceToHost));

      blocks = blocks2;
    }

    double re = 0;
    double im = 0;

    for (unsigned i = 0; i < blocks; ++i) {
      re += res2[i].re;
      im += res2[i].im;
    }

    return {mul * re, mul * im};
  }

  template <typename AQ>
  unsigned GetHighQubits(const std::vector<unsigned>& qs, unsigned qi,
                         const std::vector<unsigned>& cqs, unsigned ci,
                         unsigned ai, idx_type& cmaskh, AQ& aqs) const {
    while (1) {
      if (qi < qs.size() && (ci == cqs.size() || qs[qi] < cqs[ci])) {
        aqs[ai++] = qs[qi++];
      } else if (ci < cqs.size()) {
        cmaskh |= idx_type{1} << cqs[ci];
        aqs[ai++] = cqs[ci++];
      } else {
        break;
      }
    }

    return ai;
  }

  template <typename QS>
  void GetMs(unsigned num_qubits, const QS& qs, unsigned qs_size,
             idx_type* ms) const {
    if (qs_size == 0) {
      ms[0] = idx_type(-1);
    } else {
      idx_type xs = idx_type{1} << (qs[0] + 1);
      ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < qs_size; ++i) {
        ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs - 1);
        xs = idx_type{1} << (qs[i] + 1);
      }
      ms[qs_size] = ((idx_type{1} << num_qubits) - 1) ^ (xs - 1);
    }
  }

  template <typename QS>
  void GetXss(unsigned num_qubits, const QS& qs, unsigned qs_size,
              idx_type* xss) const {
    if (qs_size == 0) {
      xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
      }

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        xss[i] = a;
      }
    }
  }

  template <unsigned G, typename qs_type>
  void GetIndicesH(unsigned num_qubits, const qs_type& qs, unsigned qs_size,
                   IndicesH<G>& indices) const {
    if (qs_size == 0) {
      indices.ms[0] = idx_type(-1);
      indices.xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      indices.ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
        indices.ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs[i - 1] - 1);
      }
      indices.ms[g] = ((idx_type{1} << num_qubits) - 1) ^ (xs[g - 1] - 1);

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        indices.xss[i] = a;
      }
    }
  }

  template <unsigned G>
  void GetIndicesL(unsigned num_effective_qs, unsigned qmask,
                   IndicesL<G>& indices) const {
    for (unsigned i = num_effective_qs + 1; i < (G + 1); ++i) {
      indices.ms[i] = 0;
    }

    for (unsigned i = (1 << num_effective_qs); i < indices.gsize; ++i) {
      indices.xss[i] = 0;
    }

    for (unsigned i = 0; i < indices.gsize; ++i) {
      indices.qis[i] = bits::ExpandBits(i, 5 + num_effective_qs, qmask);
    }

    unsigned tmask = ((1 << (5 + num_effective_qs)) - 1) ^ qmask;
    for (unsigned i = 0; i < 32; ++i) {
      indices.tis[i] = bits::ExpandBits(i, 5 + num_effective_qs, tmask);
    }
  }

  template <unsigned G>
  unsigned GetIndicesL(unsigned num_qubits, const std::vector<unsigned>& qs,
                       IndicesL<G>& indices) const {
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits);
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    if (qs.size() == num_low_qs) {
      while (ei < num_effective_qs && l++ < num_low_qs) {
        eqs[ei] = ei + 5;
        ++ei;
      }
    } else {
      while (ei < num_effective_qs && l < num_low_qs) {
        unsigned ei5 = ei + 5;
        eqs[ei] = ei5;
        if (qi < qs.size() && qs[qi] == ei5) {
          ++qi;
          qmaskh |= 1 << ei5;
        } else {
          ++l;
        }
        ++ei;
      }

      while (ei < num_effective_qs) {
        eqs[ei] = qs[qi++];
        qmaskh |= 1 << (ei + 5);
        ++ei;
      }
    }

    GetIndicesH(num_qubits, eqs, num_effective_qs, indices);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    return num_effective_qs;
  }

  template <unsigned G>
  DataC GetIndicesLC(unsigned num_qubits, const std::vector<unsigned>& qs,
                     const std::vector<unsigned>& cqs, uint64_t cvals,
                     IndicesL<G>& indices) const {
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ci = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    while (ai < num_qubits && l < num_low_qs) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        eqs[ei++] = ai;
        qmaskh |= 1 << (ai - ci);
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        eqs[ei++] = ai;
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = qi;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);

    return {cvalsh, num_aqs, num_effective_qs};
  }

  template <unsigned G>
  DataC GetIndicesLCL(unsigned num_qubits, const std::vector<unsigned>& qs,
                      const std::vector<unsigned>& cqs, uint64_t cvals,
                      IndicesLC<G>& indices) const {
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;
    idx_type cmaskl = 0;
    idx_type cis_mask = 0;

    unsigned qi = 0;
    unsigned ci = 0;

    for (unsigned k = 0; k < 5; ++k) {
      if (qi < qs.size() && qs[qi] == k) {
        qmaskl |= 1 << (k - ci);
        ++qi;
      } else if (ci < cqs.size() && cqs[ci] == k) {
        cmaskl |= idx_type{1} << k;
        ++ci;
      }
    }

    unsigned num_low_qs = qi;
    unsigned num_low_cqs = ci;

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ei = 0;
    unsigned num_low = num_low_qs + num_low_cqs;
    unsigned remaining_low_cqs = num_low_cqs;
    unsigned effective_low_qs = num_low_qs;
    unsigned highest_cis_bit = 0;

    while (ai < num_qubits && l < num_low) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        if ((ai - ci) > 4) {
          eqs[ei++] = ai;
          qmaskh |= 1 << (ai - ci);
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          qmaskl |= 1 << (ai - ci);
          --remaining_low_cqs;
          ++effective_low_qs;
        }
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        if (remaining_low_cqs == 0) {
          eqs[ei++] = ai;
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          --remaining_low_cqs;
        }
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = effective_low_qs;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);
    idx_type cvalsl = bits::ExpandBits(idx_type(cvals), 5, cmaskl);

    cis_mask |= 31 ^ cmaskl;
    highest_cis_bit = highest_cis_bit < 5 ? 5 : highest_cis_bit;
    for (idx_type i = 0; i < 32; ++i) {
      auto c = bits::ExpandBits(i, highest_cis_bit + 1, cis_mask);
      indices.cis[i] = 2 * (c & 0xffffffe0) | (c & 0x1f) | cvalsl;
    }

    return {cvalsh, num_aqs, num_effective_qs, remaining_low_cqs};
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

  char* d_ws;
  char h_ws0[max_buf_size];
  char* h_ws = (char*) h_ws0;

  void* scratch_;
  uint64_t scratch_size_;
};

}  // namespace qsim

#endif  // SIMULATOR_CUDA_H_
