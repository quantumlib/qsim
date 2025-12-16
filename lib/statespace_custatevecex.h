// Copyright 2025 Google LLC. All Rights Reserved.
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

#ifndef STATESPACE_CUSTATEVECEX_H_
#define STATESPACE_CUSTATEVECEX_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <custatevecEx.h>

#include "multiprocess_custatevecex.h"
#include "statespace.h"
#include "util_custatevec.h"
#include "util_custatevecex.h"
#include "vectorspace_custatevecex.h"

namespace qsim {

namespace detail {

template <typename FP>
__global__ void SetStateKernel(FP v, uint64_t size, void* state) {
  uint64_t k = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;

  if (k < size) {
    ((FP*) state)[2 * k] = v;
    ((FP*) state)[2 * k + 1] = 0;
  }
}

}  // namespace detail

/**
 * Object containing context and routines for cuStateVec state-vector
 * manipulations. It is not recommended to use `GetAmpl` and `SetAmpl`.
 */
template <typename FP = float>
class StateSpaceCuStateVecEx :
    public StateSpace<StateSpaceCuStateVecEx<FP>, VectorSpaceCuStateVecEx, FP> {
 private:
  using Base =
      StateSpace<StateSpaceCuStateVecEx<FP>, VectorSpaceCuStateVecEx, FP>;

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;
  using Parameter = typename Base::Parameter;

  static constexpr auto kStateDataType = Base::kStateDataType;
  static constexpr auto kMatrixDataType = kStateDataType;
  static constexpr auto kExMatrixType = CUSTATEVEC_EX_MATRIX_DENSE;
  static constexpr auto kMatrixLayout = CUSTATEVEC_MATRIX_LAYOUT_ROW;

  explicit StateSpaceCuStateVecEx(const MultiProcessCuStateVecEx& mp,
                                  Parameter param = Parameter{})
      : Base(param, mp) {}

  static uint64_t MinSize(unsigned num_qubits) {
    return 2 * (uint64_t{1} << num_qubits);
  };

  void InternalToNormalOrder(State& state) const {
    state.to_normal_order();
  }

  void NormalToInternalOrder(State& state) const {
  }

  void SetAllZeros(State& state) const {
    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();

    auto f = [&size](unsigned i, const auto& r) {
      unsigned threads = size < 256 ? size : 256;
      unsigned blocks = size / threads;
      fp_type zero = 0.0;
      detail::SetStateKernel<<<blocks, threads>>>(zero, size, r.device_ptr);
    };

    state.assign(f);
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
    uint64_t size = uint64_t{1} << state.num_qubits();
    fp_type v = double{1} / std::sqrt(size);
    size /= state.num_substates();

    auto f = [&size, &v](unsigned i, const auto& r) {
      unsigned threads = size < 256 ? size : 256;
      unsigned blocks = size / threads;
      detail::SetStateKernel<<<blocks, threads>>>(v, size, r.device_ptr);
    };

    state.assign(f);
  }

  // |0> state.
  void SetStateZero(State& state) const {
    ErrorCheck((custatevecExStateVectorSetZeroState(state.get())));
  }

  // It is not recommended to use this function.
  std::complex<fp_type> GetAmpl(const State& state, uint64_t i) const {
    fp_type buf[2] = {0, 0};

    uint64_t k = 0;
    const auto& wire_ordering = state.get_wire_ordering();
    for (unsigned j = 0; j < state.num_qubits(); ++j) {
      k |= ((i >>  wire_ordering[j]) & 1) << j;
    }

    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();
    unsigned required_rank = k / size;

    if (state.distr_type() != Base::kMultiProcess
        || Base::mp.rank() == required_rank) {
      ErrorCheck(custatevecExStateVectorGetState(
          state.get(), buf, kStateDataType, k, k + 1, 1));
    }

    ErrorCheck(custatevecExStateVectorSynchronize(state.get()));

    if (state.distr_type() == Base::kMultiProcess) {
      auto cuda_type = GetCudaType<std::complex<fp_type>>();
      auto comm = Base::mp.communicator();
      ErrorCheck(comm->intf->bcast(comm, buf, 1, cuda_type, required_rank));
    }

    return {buf[0], buf[1]};
  }

  // It is not recommended to use this function.
  void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) const {
    fp_type buf[2] = {std::real(ampl), std::imag(ampl)};

    uint64_t k = 0;
    const auto& wire_ordering = state.get_wire_ordering();
    for (unsigned j = 0; j < state.num_qubits(); ++j) {
      k |= ((i >>  wire_ordering[j]) & 1) << j;
    }

    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();
    unsigned required_rank = k / size;

    if (state.distr_type() != Base::kMultiProcess
        || Base::mp.rank() == required_rank) {
      ErrorCheck(custatevecExStateVectorSetState(
          state.get(), buf, kStateDataType, k, k + 1, 1));
    }

    ErrorCheck(custatevecExStateVectorSynchronize(state.get()));
  }

  // It is not recommended to use this function.
  void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) const {
    fp_type buf[2] = {re, im};

    uint64_t k = 0;
    const auto& wire_ordering = state.get_wire_ordering();
    for (unsigned j = 0; j < state.num_qubits(); ++j) {
      k |= ((i >>  wire_ordering[j]) & 1) << j;
    }

    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();
    unsigned required_rank = k / size;

    if (state.distr_type() != Base::kMultiProcess
        || Base::mp.rank() == required_rank) {
      ErrorCheck(custatevecExStateVectorSetState(
          state.get(), buf, kStateDataType, k, k + 1, 1));
    }

    ErrorCheck(custatevecExStateVectorSynchronize(state.get()));
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  static void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                          const std::complex<fp_type>& val,
                          bool exclude = false) {
    // Not implemented.
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  static void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                          fp_type im, bool exclude = false) {
    // Not implemented.
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    uint64_t size = (uint64_t{1} << src.num_qubits()) / src.num_substates();

    auto f = [&size](unsigned i, const auto& rd, const auto& rs) {
      cublasHandle_t cublas_handle;
      ErrorCheck(cublasCreate(&cublas_handle));
      ErrorCheck(cublasSetStream(cublas_handle, rd.stream));

      if (Base::is_float) {
        cuComplex a = {1.0, 0.0};
        auto p1 = (const cuComplex*) rs.device_ptr;
        auto p2 = (cuComplex*) rd.device_ptr;
        ErrorCheck(cublasCaxpy(cublas_handle, size, &a, p1, 1, p2, 1));
      } else {
        cuDoubleComplex a = {1.0, 0.0};
        auto p1 = (const cuDoubleComplex*) rs.device_ptr;
        auto p2 = (cuDoubleComplex*) rd.device_ptr;
        ErrorCheck(cublasZaxpy(cublas_handle, size, &a, p1, 1, p2, 1));
      }

      ErrorCheck(cudaStreamSynchronize(rd.stream));
      ErrorCheck(cublasDestroy(cublas_handle));
    };

    dest.assign(src, f);

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  static void Multiply(fp_type a, State& state) {
    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();

    auto f = [&a, &size](unsigned i, const auto& r) {
      cublasHandle_t cublas_handle;
      ErrorCheck(cublasCreate(&cublas_handle));
      ErrorCheck(cublasSetStream(cublas_handle, r.stream));

      if (Base::is_float) {
        float a1 = a;
        auto p = (cuComplex*) r.device_ptr;
        ErrorCheck(cublasCsscal(cublas_handle, size, &a1, p, 1));
      } else {
        double a1 = a;
        auto p = (cuDoubleComplex*) r.device_ptr;
        ErrorCheck(cublasZdscal(cublas_handle, size, &a1, p, 1));
      }

      ErrorCheck(cudaStreamSynchronize(r.stream));
      ErrorCheck(cublasDestroy(cublas_handle));
    };

    return state.assign(f);
  }

  // Does the equivalent of state *= (re + i im) elementwise.
  static void Multiply(fp_type re, fp_type im, State& state) {
    uint64_t size = (uint64_t{1} << state.num_qubits()) / state.num_substates();

    auto f = [&re, &im, &size](unsigned i, const auto& r) {
      cublasHandle_t cublas_handle;
      ErrorCheck(cublasCreate(&cublas_handle));
      ErrorCheck(cublasSetStream(cublas_handle, r.stream));

      if (Base::is_float) {
        cuComplex a = {float(re), float(im)};
        auto p = (cuComplex*) r.device_ptr;
        ErrorCheck(cublasCscal(cublas_handle, size, &a, p, 1));
      } else {
        cuDoubleComplex a = {re, im};
        auto p = (cuDoubleComplex*) r.device_ptr;
        ErrorCheck(cublasZscal(cublas_handle, size, &a, p, 1));
      }

      ErrorCheck(cudaStreamSynchronize(r.stream));
      ErrorCheck(cublasDestroy(cublas_handle));
    };

    return state.assign(f);
  }

  static std::complex<double> InnerProduct(
      const State& state1, const State& state2) {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    uint64_t size =
        (uint64_t{1} << state1.num_qubits()) / state1.num_substates();

    auto f = [&size](unsigned i, const auto& r1, const auto& r2) {
      cublasHandle_t cublas_handle;
      ErrorCheck(cublasCreate(&cublas_handle));
      ErrorCheck(cublasSetStream(cublas_handle, r1.stream));

      if (Base::is_float) {
        cuComplex result;
        auto p1 = (const cuComplex*) r1.device_ptr;
        auto p2 = (const cuComplex*) r2.device_ptr;
        ErrorCheck(cublasCdotc(cublas_handle, size, p1, 1, p2, 1, &result));
        return std::complex<double>{cuCrealf(result), cuCimagf(result)};
      } else {
        cuDoubleComplex result;
        auto p1 = (const cuDoubleComplex*) r1.device_ptr;
        auto p2 = (const cuDoubleComplex*) r2.device_ptr;
        ErrorCheck(cublasZdotc(cublas_handle, size, p1, 1, p2, 1, &result));
        return std::complex<double>{cuCreal(result), cuCimag(result)};
      }

      ErrorCheck(cudaStreamSynchronize(r1.stream));
      ErrorCheck(cublasDestroy(cublas_handle));
    };

    return state1.reduce(state2, f);
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    return std::real(InnerProduct(state1, state2));
  }

  double Norm(const State& state) const {
    double norm;

    ErrorCheck(custatevecExAbs2SumArray(
        state.get(), &norm, nullptr, 0, nullptr, nullptr, 0));
    ErrorCheck(custatevecExStateVectorSynchronize(state.get()));

    return norm;
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;

    if (num_samples > 0) {
      auto rs = GenerateRandomValues<double>(num_samples, seed, 1.0);

      std::vector<custatevecIndex_t> bitstrings0(num_samples);

      std::vector<int32_t> wires;
      wires.reserve(state.num_qubits());
      for (unsigned i = 0; i < state.num_qubits(); ++i) {
        wires[i] = i;
      }

      ErrorCheck(custatevecExSample(
          state.get(), bitstrings0.data(), wires.data(), state.num_qubits(),
          rs.data(), num_samples, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER,
          nullptr));
      ErrorCheck(custatevecExStateVectorSynchronize(state.get()));

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

    custatevecIndex_t bits;

    ErrorCheck(custatevecExMeasure(
        state.get(), &bits, (int32_t*) qubits.data(), qubits.size(),
        r, collapse, nullptr));
    ErrorCheck(custatevecExStateVectorSynchronize(state.get()));

    for (std::size_t i = 0; i < qubits.size(); ++i) {
      uint64_t bit = (bits >> i) & 1;
      result.bitstring[i] = bit;
      result.bits |= bit << qubits[i];
    }

    return result;
  }

  template <typename RGen>
  MeasurementResult VirtualMeasure(const std::vector<unsigned>& qubits,
                                   RGen& rgen, const State& state) const {
    return Measure(qubits, rgen, const_cast<State&>(state), true);
  }

  void Collapse(const MeasurementResult& mr, State& state) const {
    // Not implemented.
  }
};

}  // namespace qsim

#endif  // STATESPACE_CUSTATEVECEX_H_
