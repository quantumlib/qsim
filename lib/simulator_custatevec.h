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

#ifndef SIMULATOR_CUSTATEVEC_H_
#define SIMULATOR_CUSTATEVEC_H_

#include <complex>
#include <cstdint>
#include <type_traits>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <custatevec.h>

#include "io.h"
#include "statespace_custatevec.h"
#include "util_custatevec.h"

namespace qsim {

/**
 * Quantum circuit simulator using the NVIDIA cuStateVec library.
 */
template <typename FP = float>
class SimulatorCuStateVec final {
 public:
  using StateSpace = StateSpaceCuStateVec<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  static constexpr auto kStateType = StateSpace::kStateType;
  static constexpr auto kMatrixType = StateSpace::kMatrixType;
  static constexpr auto kExpectType = StateSpace::kExpectType;
  static constexpr auto kComputeType = StateSpace::kComputeType;
  static constexpr auto kMatrixLayout = StateSpace::kMatrixLayout;

  explicit SimulatorCuStateVec(const cublasHandle_t& cublas_handle,
                               const custatevecHandle_t& custatevec_handle)
      : cublas_handle_(cublas_handle), custatevec_handle_(custatevec_handle),
      workspace_(nullptr), workspace_size_(0) {}

  ~SimulatorCuStateVec() {
    ErrorCheck(cudaFree(workspace_));
  }

  /**
   * Applies a gate using the NVIDIA cuStateVec library.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    if (qs.size() == 0) {
      uint64_t size = uint64_t{1} << state.num_qubits();

      if (StateSpace::is_float) {
        cuComplex a = {(float) matrix[0], (float) matrix[1]};
        auto p = (cuComplex*) state.get();
        ErrorCheck(cublasCscal(cublas_handle_, size, &a, p, 1));
      } else {
        cuDoubleComplex a = {matrix[0], matrix[1]};
        auto p = (cuDoubleComplex*) state.get();
        ErrorCheck(cublasZscal(cublas_handle_, size, &a, p, 1));
      }
    } else {
      auto workspace_size = ApplyGateWorkSpaceSize(
          state.num_qubits(), qs.size(), 0, matrix);
      AllocWorkSpace(workspace_size);

      ErrorCheck(custatevecApplyMatrix(
                     custatevec_handle_, state.get(), kStateType,
                     state.num_qubits(), matrix, kMatrixType, kMatrixLayout, 0,
                     (int32_t*) qs.data(), qs.size(), nullptr, nullptr, 0,
                     kComputeType, workspace_, workspace_size));
    }
  }

  /**
   * Applies a controlled gate using the NVIDIA cuStateVec library.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, State& state) const {
    if (qs.size() == 0) {
      IO::errorf(
          "error: controlled global phase gate is not implemented %s %d\n",
          __FILE__, __LINE__);
      exit(1);
    } else {
      std::vector<int32_t> control_bits;
      control_bits.reserve(cqs.size());

      for (std::size_t i = 0; i < cqs.size(); ++i) {
        control_bits.push_back((cmask >> i) & 1);
      }

      auto workspace_size = ApplyGateWorkSpaceSize(
          state.num_qubits(), qs.size(), cqs.size(), matrix);
      AllocWorkSpace(workspace_size);

      ErrorCheck(custatevecApplyMatrix(
                     custatevec_handle_, state.get(), kStateType,
                     state.num_qubits(), matrix, kMatrixType, kMatrixLayout, 0,
                     (int32_t*) qs.data(), qs.size(),
                     (int32_t*) cqs.data(), control_bits.data(), cqs.size(),
                     kComputeType, workspace_, workspace_size));
    }
  }

  /**
   * Computes the expectation value of an operator using the NVIDIA cuStateVec
   * library.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    auto workspace_size = ExpectationValueWorkSpaceSize(
        state.num_qubits(), qs.size(), matrix);
    AllocWorkSpace(workspace_size);

    cuDoubleComplex eval;

    ErrorCheck(custatevecComputeExpectation(
                   custatevec_handle_, state.get(), kStateType,
                   state.num_qubits(), &eval, kExpectType, nullptr, matrix,
                   kMatrixType, kMatrixLayout, (int32_t*) qs.data(), qs.size(),
                   kComputeType, workspace_, workspace_size));

    return {cuCreal(eval), cuCimag(eval)};
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 32;
  }

 private:
  size_t ApplyGateWorkSpaceSize(
      unsigned num_qubits, unsigned num_targets, unsigned num_controls,
      const fp_type* matrix) const {
    size_t size;

    ErrorCheck(custatevecApplyMatrixGetWorkspaceSize(
                   custatevec_handle_, kStateType, num_qubits, matrix,
                   kMatrixType, kMatrixLayout, 0, num_targets, num_controls,
                   kComputeType, &size));

    return size;
  }

  size_t ExpectationValueWorkSpaceSize(
      unsigned num_qubits, unsigned num_targets, const fp_type* matrix) const {
    size_t size;

    ErrorCheck(custatevecComputeExpectationGetWorkspaceSize(
                   custatevec_handle_, kStateType, num_qubits, matrix,
                   kMatrixType, kMatrixLayout, num_targets, kComputeType,
                   &size));

    return size;
  }

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

#endif  // SIMULATOR_CUSTATEVEC_H_
