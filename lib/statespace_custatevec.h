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

#ifndef STATESPACE_CUSTATEVEC_H_
#define STATESPACE_CUSTATEVEC_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <custatevec.h>

#include "statespace.h"
#include "util_custatevec.h"
#include "vectorspace_cuda.h"

namespace qsim {

namespace detail {

template <typename FP>
__global__ void SetStateUniformKernel(FP v, uint64_t size, FP* state) {
  uint64_t k = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;

  if (k < size) {
    state[2 * k] = v;
    state[2 * k + 1] = 0;
  }
}

}  // namespace detail

/**
 * Object containing context and routines for cuStateVec state-vector
 * manipulations. It is not recommended to use `GetAmpl` and `SetAmpl`.
 */
template <typename FP = float>
class StateSpaceCuStateVec :
    public StateSpace<StateSpaceCuStateVec<FP>, VectorSpaceCUDA, FP> {
 private:
  using Base = StateSpace<StateSpaceCuStateVec<FP>, qsim::VectorSpaceCUDA, FP>;

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

 private:
  static constexpr auto is_float = std::is_same<fp_type, float>::value;

 public:
  static constexpr auto kStateType = is_float ? CUDA_C_32F : CUDA_C_64F;
  static constexpr auto kMatrixType = kStateType;
  static constexpr auto kExpectType = CUDA_C_64F;
  static constexpr auto kComputeType =
      is_float ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;
  static constexpr auto kMatrixLayout = CUSTATEVEC_MATRIX_LAYOUT_ROW;

  explicit StateSpaceCuStateVec(const cublasHandle_t& cublas_handle,
                                const custatevecHandle_t& custatevec_handle)
      : cublas_handle_(cublas_handle), custatevec_handle_(custatevec_handle),
        workspace_(nullptr), workspace_size_(0) {}

  virtual ~StateSpaceCuStateVec() {
    if (workspace_ != nullptr) {
      ErrorCheck(cudaFree(workspace_));
    }
  }

  static uint64_t MinSize(unsigned num_qubits) {
    return 2 * (uint64_t{1} << num_qubits);
  };

  void InternalToNormalOrder(State& state) const {
  }

  void NormalToInternalOrder(State& state) const {
  }

  void SetAllZeros(State& state) const {
    ErrorCheck(cudaMemset(state.get(), 0,
                          MinSize(state.num_qubits()) * sizeof(fp_type)));
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    uint64_t size = uint64_t{1} << state.num_qubits();

    unsigned threads = size < 256 ? size : 256;
    unsigned blocks = size / threads;

    fp_type v = double{1} / std::sqrt(size);

    detail::SetStateUniformKernel<<<blocks, threads>>>(v, size, state.get());
    ErrorCheck(cudaPeekAtLastError());
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
    fp_type a[2];
    auto p = state.get() + 2 * i;
    ErrorCheck(cudaMemcpy(a, p, 2 * sizeof(fp_type), cudaMemcpyDeviceToHost));
    return std::complex<fp_type>(a[0], a[1]);
  }

  // It is not recommended to use this function.
  static void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    fp_type a[2] = {std::real(ampl), std::imag(ampl)};
    auto p = state.get() + 2 * i;
    ErrorCheck(cudaMemcpy(p, a, 2 * sizeof(fp_type), cudaMemcpyHostToDevice));
  }

  // It is not recommended to use this function.
  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
    fp_type a[2] = {re, im};
    auto p = state.get() + 2 * i;
    ErrorCheck(cudaMemcpy(p, a, 2 * sizeof(fp_type), cudaMemcpyHostToDevice));
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val,
                   bool exclude = false) const {
    // Not implemented.
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im, bool exclude = false) const {
    // Not implemented.
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    uint64_t size = uint64_t{1} << src.num_qubits();

    if (is_float) {
      cuComplex a = {1.0, 0.0};
      auto p1 = (const cuComplex*) src.get();
      auto p2 = (cuComplex*) dest.get();
      ErrorCheck(cublasCaxpy(cublas_handle_, size, &a, p1, 1, p2, 1));
    } else {
      cuDoubleComplex a = {1.0, 0.0};
      auto p1 = (const cuDoubleComplex*) src.get();
      auto p2 = (cuDoubleComplex*) dest.get();
      ErrorCheck(cublasZaxpy(cublas_handle_, size, &a, p1, 1, p2, 1));
    }

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  void Multiply(fp_type a, State& state) const {
    uint64_t size = uint64_t{1} << state.num_qubits();

    if (is_float) {
      float a1 = a;
      auto p = (cuComplex*) state.get();
      ErrorCheck(cublasCsscal(cublas_handle_, size, &a1, p, 1));
    } else {
      double a1 = a;
      auto p = (cuDoubleComplex*) state.get();
      ErrorCheck(cublasZdscal(cublas_handle_, size, &a1, p, 1));
    }
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    uint64_t size = uint64_t{1} << state1.num_qubits();

    if (is_float) {
      cuComplex result;
      auto p1 = (const cuComplex*) state1.get();
      auto p2 = (const cuComplex*) state2.get();
      ErrorCheck(cublasCdotc(cublas_handle_, size, p1, 1, p2, 1, &result));
      return {cuCrealf(result), cuCimagf(result)};
    } else {
      cuDoubleComplex result;
      auto p1 = (const cuDoubleComplex*) state1.get();
      auto p2 = (const cuDoubleComplex*) state2.get();
      ErrorCheck(cublasZdotc(cublas_handle_, size, p1, 1, p2, 1, &result));
      return {cuCreal(result), cuCimag(result)};
    }
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    return std::real(InnerProduct(state1, state2));
  }

  double Norm(const State& state) const {
    uint64_t size = uint64_t{1} << state.num_qubits();

    if (is_float) {
      float result;
      auto p = (const cuComplex*) state.get();
      ErrorCheck(cublasScnrm2(cublas_handle_, size, p, 1, &result));
      return result * result;
    } else {
      double result;
      auto p = (const cuDoubleComplex*) state.get();
      ErrorCheck(cublasDznrm2(cublas_handle_, size, p, 1, &result));
      return result * result;
    }
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      auto rs = GenerateRandomValues<double>(num_samples, seed, 1.0);

      size_t workspace_size;
      custatevecSamplerDescriptor_t sampler;

      ErrorCheck(custatevecSamplerCreate(
                     custatevec_handle_, state.get(), kStateType,
                     state.num_qubits(), &sampler, num_samples,
                     &workspace_size));

      AllocWorkSpace(workspace_size);

      ErrorCheck(custatevecSamplerPreprocess(
                     custatevec_handle_, sampler, workspace_, workspace_size));

      std::vector<custatevecIndex_t> bitstrings0(num_samples);
      std::vector<int32_t> bitordering;

      bitordering.reserve(state.num_qubits());
      for (unsigned i = 0; i < state.num_qubits(); ++i) {
        bitordering.push_back(i);
      }

      ErrorCheck(custatevecSamplerSample(
                     custatevec_handle_, sampler, bitstrings0.data(),
                     bitordering.data(), state.num_qubits(), rs.data(),
                     num_samples, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));

      bitstrings.reserve(num_samples);
      for (unsigned i = 0; i < num_samples; ++i) {
        bitstrings.push_back(bitstrings0[i]);
      }
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  template <typename RGen>
  MeasurementResult Measure(const std::vector<unsigned>& qubits,
                            RGen& rgen, State& state,
                            bool no_collapse = false) const {
    auto r = RandomValue(rgen, 1.0);

    MeasurementResult result;

    result.valid = true;
    result.mask = 0;
    result.bits = 0;
    result.bitstring.resize(qubits.size(), 0);

    for (auto q : qubits) {
      if (q >= state.num_qubits()) {
        result.valid = false;
        return result;
      }

      result.mask |= uint64_t{1} << q;
    }

    auto collapse = no_collapse ?
        CUSTATEVEC_COLLAPSE_NONE : CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO;

    ErrorCheck(custatevecBatchMeasure(
                   custatevec_handle_, state.get(), kStateType,
                   state.num_qubits(), (int*) result.bitstring.data(),
                   (int*) qubits.data(), qubits.size(), r, collapse));

    for (std::size_t i = 0; i < result.bitstring.size(); ++i) {
      result.bits |= result.bitstring[i] << qubits[i];
    }

    return result;
  }

  template <typename RGen>
  MeasurementResult VirtualMeasure(const std::vector<unsigned>& qubits,
                                   RGen& rgen, const State& state) const {
    return Measure(qubits, rgen, const_cast<State&>(state), true);
  }

  void Collapse(const MeasurementResult& mr, State& state) const {
    unsigned count = 0;

    std::vector<int> bitstring;
    std::vector<int> bitordering;

    bitstring.reserve(state.num_qubits());
    bitordering.reserve(state.num_qubits());

    for (unsigned i = 0; i < state.num_qubits(); ++i) {
      if (((mr.mask >> i) & 1) != 0) {
        bitstring.push_back((mr.bits >> i) & 1);
        bitordering.push_back(i);
        ++count;
      }
    }

    ErrorCheck(custatevecCollapseByBitString(
                   custatevec_handle_, state.get(), kStateType,
                   state.num_qubits(), bitstring.data(), bitordering.data(),
                   count, 1.0));

    // TODO: do we need the following?
    double norm = Norm(state);
    Multiply(1.0 / std::sqrt(norm), state);
  }

 private:
  void* AllocWorkSpace(size_t size) const {
    if (size > workspace_size_) {
      if (workspace_ != nullptr) {
        ErrorCheck(cudaFree(workspace_));
      }

      ErrorCheck(cudaMalloc(const_cast<void**>(&workspace_), size));

      const_cast<uint64_t&>(workspace_size_) = size;
    }

    return workspace_;
  }

  const cublasHandle_t cublas_handle_;
  const custatevecHandle_t custatevec_handle_;

  void* workspace_;
  size_t workspace_size_;
};

}  // namespace qsim

#endif  // STATESPACE_CUSTATEVEC_H_
