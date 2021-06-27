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

#ifndef STATESPACE_CUDA_H_
#define STATESPACE_CUDA_H_

#include <cuda.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#include "statespace.h"
#include "statespace_cuda_kernels.h"
#include "vectorspace_cuda.h"
#include "util_cuda.h"

namespace qsim {

/**
 * Object containing context and routines for CUDA state-vector manipulations.
 * State is a vectorized sequence of 32 real components followed by 32
 * imaginary components. 32 floating numbers can be proccessed in parallel by
 * a single warp. It is not recommended to use `GetAmpl` and `SetAmpl`.
 */
template <typename FP = float>
class StateSpaceCUDA :
    public StateSpace<StateSpaceCUDA<FP>, VectorSpaceCUDA, FP> {
 private:
  using Base = StateSpace<StateSpaceCUDA<FP>, qsim::VectorSpaceCUDA, FP>;

 protected:
  struct Grid {
    unsigned threads;
    unsigned dblocks;
    unsigned blocks;
  };

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  struct Parameter {
    /**
     * The number of threads per block.
     * Should be 2 to the power of k, where k is in the range [5,10].
     */
    unsigned num_threads = 512;
    /**
     * The number of data blocks. Each thread processes num_dblocks data
     * blocks in reductions (norms, inner products, etc).
     */
    unsigned num_dblocks = 16;
  };

  explicit StateSpaceCUDA(const Parameter& param)
      : param_(param), scratch_(nullptr), scratch_size_(0) {}

  virtual ~StateSpaceCUDA() {
    if (scratch_ != nullptr) {
      ErrorCheck(cudaFree(scratch_));
    }
  }

  static uint64_t MinSize(unsigned num_qubits) {
    return std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
  };

  void InternalToNormalOrder(State& state) const {
    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;
    unsigned bytes = 2 * threads * sizeof(fp_type);

    InternalToNormalOrderKernel<<<blocks, threads, bytes>>>(state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void NormalToInternalOrder(State& state) const {
    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;
    unsigned bytes = 2 * threads * sizeof(fp_type);

    NormalToInternalOrderKernel<<<blocks, threads, bytes>>>(state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  void SetAllZeros(State& state) const {
    cudaMemset(state.get(), 0, MinSize(state.num_qubits()) * sizeof(fp_type));
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    uint64_t size = MinSize(state.num_qubits()) / 2;
    uint64_t hsize = uint64_t{1} << state.num_qubits();

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    fp_type v = double{1} / std::sqrt(hsize);

    SetStateUniformKernel<<<blocks, threads>>>(v, hsize, state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  // |0> state.
  void SetStateZero(State& state) const {
    SetAllZeros(state);
    fp_type one[1] = {1};
    ErrorCheck(
        cudaMemcpy(state.get(), one, sizeof(fp_type), cudaMemcpyHostToDevice));
  }

  // It is not recommended to use this function.
  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    fp_type re, im;
    auto p = state.get() + 64 * (i / 32) + i % 32;
    ErrorCheck(cudaMemcpy(&re, p, sizeof(fp_type), cudaMemcpyDeviceToHost));
    ErrorCheck(
        cudaMemcpy(&im, p + 32, sizeof(fp_type), cudaMemcpyDeviceToHost));
    return std::complex<fp_type>(re, im);
  }

  // It is not recommended to use this function.
  static void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    fp_type re = std::real(ampl);
    fp_type im = std::imag(ampl);
    auto p = state.get() + 64 * (i / 32) + i % 32;
    ErrorCheck(cudaMemcpy(p, &re, sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(p + 32, &im, sizeof(fp_type), cudaMemcpyHostToDevice));
  }

  // It is not recommended to use this function.
  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
    auto p = state.get() + 64 * (i / 32) + i % 32;
    ErrorCheck(cudaMemcpy(p, &re, sizeof(fp_type), cudaMemcpyHostToDevice));
    ErrorCheck(
        cudaMemcpy(p + 32, &im, sizeof(fp_type), cudaMemcpyHostToDevice));
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val,
                   bool exclude = false) const {
    BulkSetAmpl(state, mask, bits, std::real(val), std::imag(val), exclude);
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im, bool exclude = false) const {
    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    BulkSetAmplKernel<<<blocks, threads>>>(
        mask, bits, re, im, exclude, state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    uint64_t size = MinSize(src.num_qubits());

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    AddKernel<<<blocks, threads>>>(src.get(), dest.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  void Multiply(fp_type a, State& state) const {
    uint64_t size = MinSize(state.num_qubits());

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    MultiplyKernel<<<blocks, threads>>>(a, state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    using C = Complex<double>;
    auto r = Reduce<C, C, Product<fp_type>>(state1, state2);

    return {r.re, r.im};
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    return Reduce<double, double, RealProduct<fp_type>>(state1, state2);
  }

  double Norm(const State& state) const {
    return Reduce<double, double, RealProduct<fp_type>>(state, state);
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      Grid g1 = GetGrid1(MinSize(state.num_qubits()) / 2);
      unsigned bytes = g1.threads * sizeof(double);

      unsigned scratch_size = (g1.blocks + 1) * sizeof(double)
          + num_samples * (sizeof(uint64_t) + sizeof(DistrRealType));

      void* scratch = AllocScratch(scratch_size);

      double* d_res2 = (double*) scratch;
      double* d_res1 = d_res2 + 1;
      uint64_t* d_bitstrings = (uint64_t*) (d_res1 + g1.blocks);
      DistrRealType* d_rs = (DistrRealType *) (d_bitstrings + num_samples);

      auto op1 = RealProduct<fp_type>();
      auto op2 = Plus<double>();

      Reduce1Kernel<double><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, op1, op2, op2, state.get(), state.get(), d_res1);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      double norm;

      if (g1.blocks == 1) {
        ErrorCheck(
            cudaMemcpy(&norm, d_res1, sizeof(double), cudaMemcpyDeviceToHost));
      } else {
        Grid g2 = GetGrid2(g1.blocks);
        unsigned bytes = g2.threads * sizeof(double);

        auto op3 = Plus<double>();

        Reduce2Kernel<double><<<g2.blocks, g2.threads, bytes>>>(
            g2.dblocks, g1.blocks, op3, op3, d_res1, d_res2);
        ErrorCheck(cudaPeekAtLastError());
        ErrorCheck(cudaDeviceSynchronize());

        ErrorCheck(
            cudaMemcpy(&norm, d_res2, sizeof(double), cudaMemcpyDeviceToHost));
      }

      // TODO: generate random values on the device.
      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      ErrorCheck(cudaMemcpy(d_rs, rs.data(),
                            num_samples * sizeof(DistrRealType),
                            cudaMemcpyHostToDevice));

      SampleKernel<<<1, g1.threads>>>(g1.blocks, g1.dblocks, num_samples,
                                      d_rs, d_res1, state.get(), d_bitstrings);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      bitstrings.resize(num_samples, 0);

      ErrorCheck(cudaMemcpy(bitstrings.data(), d_bitstrings,
                            num_samples * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  void Collapse(const MeasurementResult& mr, State& state) const {
    using Op = RealProduct<fp_type>;
    double r = Reduce<double, double, Op>(mr.mask, mr.bits, state, state);
    fp_type renorm = 1 / std::sqrt(r);

    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    CollapseKernel<<<blocks, threads>>>(mr.mask, mr.bits, renorm, state.get());
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());
  }

  std::vector<double> PartialNorms(const State& state) const {
    Grid g = GetGrid1(MinSize(state.num_qubits()) / 2);

    unsigned scratch_size = g.blocks * sizeof(double);
    unsigned bytes = g.threads * sizeof(double);

    double* d_res = (double*) AllocScratch(scratch_size);

    auto op1 = RealProduct<fp_type>();
    auto op2 = Plus<double>();

    Reduce1Kernel<double><<<g.blocks, g.threads, bytes>>>(
        g.dblocks, op1, op2, op2, state.get(), state.get(), d_res);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    std::vector<double> norms(g.blocks);

    ErrorCheck(
        cudaMemcpy(norms.data(), d_res, scratch_size, cudaMemcpyDeviceToHost));

    return norms;
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    Grid g = GetGrid1(MinSize(state.num_qubits()) / 2);

    uint64_t res;
    uint64_t* d_res = (uint64_t*) AllocScratch(sizeof(uint64_t));

    FindMeasuredBitsKernel<<<1, g.threads>>>(
        m, g.dblocks, r, state.get(), d_res);
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    ErrorCheck(
        cudaMemcpy(&res, d_res, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    return res & mask;
  }

 protected:
  Parameter param_;

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

  Grid GetGrid1(uint64_t size) const {
    Grid grid;

    grid.threads = std::min(size, uint64_t{param_.num_threads});
    grid.dblocks = std::min(size / grid.threads, uint64_t{param_.num_dblocks});
    grid.blocks = size / (grid.threads * grid.dblocks);

    return grid;
  }

  Grid GetGrid2(unsigned size) const {
    Grid grid;

    grid.threads = std::min(param_.num_threads, std::max(32U, size));
    grid.dblocks = std::max(1U, size / grid.threads);
    grid.blocks = 1;

    return grid;
  }

  template <typename FP1, typename FP2, typename Op>
  FP2 Reduce(const State& state1, const State& state2) const {
    return Reduce<FP1, FP2, Op>(0, 0, state1, state2);
  }

  template <typename FP1, typename FP2, typename Op>
  FP2 Reduce(uint64_t mask, uint64_t bits,
             const State& state1, const State& state2) const {
    uint64_t size = MinSize(state1.num_qubits()) / 2;

    Grid g1 = GetGrid1(size);
    unsigned bytes = g1.threads * sizeof(FP1);

    FP2* d_res2 = (FP2*) AllocScratch((g1.blocks + 1) * sizeof(FP2));
    FP2* d_res1 = d_res2 + 1;

    auto op1 = Op();
    auto op2 = Plus<FP1>();
    auto op3 = Plus<typename Scalar<FP1>::type>();

    if (mask == 0) {
      Reduce1Kernel<FP1><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, op1, op2, op3, state1.get(), state2.get(), d_res1);
    } else {
      Reduce1MaskedKernel<FP1><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, mask, bits, op1, op2, op3, state1.get(), state2.get(),
          d_res1);
    }
    ErrorCheck(cudaPeekAtLastError());
    ErrorCheck(cudaDeviceSynchronize());

    FP2 result;

    if (g1.blocks == 1) {
      ErrorCheck(
          cudaMemcpy(&result, d_res1, sizeof(FP2), cudaMemcpyDeviceToHost));
    } else {
      Grid g2 = GetGrid2(g1.blocks);
      unsigned bytes = g2.threads * sizeof(FP2);

      auto op2 = Plus<FP2>();
      auto op3 = Plus<typename Scalar<FP2>::type>();

      Reduce2Kernel<FP2><<<g2.blocks, g2.threads, bytes>>>(
          g2.dblocks, g1.blocks, op2, op3, d_res1, d_res2);
      ErrorCheck(cudaPeekAtLastError());
      ErrorCheck(cudaDeviceSynchronize());

      ErrorCheck(
          cudaMemcpy(&result, d_res2, sizeof(FP2), cudaMemcpyDeviceToHost));
    }

    return result;
  }

 private:
  void* scratch_;
  uint64_t scratch_size_;
};

}  // namespace qsim

#endif  // STATESPACE_CUDA_H_
