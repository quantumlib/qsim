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

#ifndef SIMULATOR_CUSTATEVECEX_H_
#define SIMULATOR_CUSTATEVECEX_H_

#include <complex>
#include <cstdint>
#include <type_traits>

#include <cuComplex.h>
#include <custatevecEx.h>

#include "io.h"
#include "statespace_custatevecex.h"
#include "util_custatevec.h"
#include "util_custatevecex.h"

namespace qsim {

/**
 * Quantum circuit simulator using the NVIDIA cuStateVec library.
 */
template <typename FP = float>
class SimulatorCuStateVecEx final {
 public:
  using StateSpace = StateSpaceCuStateVecEx<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  static constexpr auto kStateDataType = StateSpace::kStateDataType;
  static constexpr auto kMatrixDataType = StateSpace::kMatrixDataType;
  static constexpr auto kExMatrixType = StateSpace::kExMatrixType;
  static constexpr auto kMatrixLayout = StateSpace::kMatrixLayout;
  static constexpr auto kExpectDataType = CUDA_C_64F;
  static constexpr auto kComputeType =
      StateSpace::is_float ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;

  SimulatorCuStateVecEx() {}

  /**
   * Applies a gate using the NVIDIA cuStateVec library.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    if (qs.size() == 0) {
      StateSpace::Multiply(matrix[0], matrix[1], state);
    } else {
      unsigned num_qubits = state.num_qubits();
      unsigned num_global_qubits = get_num_global_qubits(state.num_substates());
      unsigned num_local_qubits = num_qubits - num_global_qubits;

      if (qs.size() > num_local_qubits) {
        IO::errorf("error: the number of gate qubits exceeds the number of "
                   "local qubits at %s %d.\n", __FILE__, __LINE__);
        exit(1);
      }

      ErrorCheck(custatevecExApplyMatrix(
          state.get(), matrix, kMatrixDataType, kExMatrixType, kMatrixLayout,
          0, (int32_t*) qs.data(), qs.size(), nullptr, nullptr, 0));
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
          "error: controlled global phase gate is not implemented %s %d.\n",
          __FILE__, __LINE__);
      exit(1);
    } else {
      unsigned num_qubits = state.num_qubits();
      unsigned num_global_qubits = get_num_global_qubits(state.num_substates());
      unsigned num_local_qubits = num_qubits - num_global_qubits;

      if (qs.size() > num_local_qubits) {
        IO::errorf("error: the number of gate qubits exceeds the number of "
                   "local qubits at %s %d.\n", __FILE__, __LINE__);
        exit(1);
      }

      std::vector<int32_t> control_bits;
      control_bits.reserve(cqs.size());

      for (std::size_t i = 0; i < cqs.size(); ++i) {
        control_bits.push_back((cmask >> i) & 1);
      }

      ErrorCheck(custatevecExApplyMatrix(
          state.get(), matrix, kMatrixDataType, kExMatrixType, kMatrixLayout,
          0, (int32_t*) qs.data(), qs.size(), (int32_t*) cqs.data(),
          control_bits.data(), cqs.size()));
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
    unsigned num_qubits = state.num_qubits();
    unsigned num_global_qubits = get_num_global_qubits(state.num_substates());
    unsigned num_local_qubits = num_qubits - num_global_qubits;

    if (qs.size() > num_local_qubits) {
      IO::errorf("error: the number of gate qubits exceeds the number of "
                 "local qubits at %s %d.\n", __FILE__, __LINE__);
      exit(1);
    }

    const auto& wire_ordering = state.get_wire_ordering();

    // Wire ordering can be arbitrary. The following lines make qs consistent
    // with wire ordering and permute bits if necessary.

    std::vector<unsigned> perm;
    perm.reserve(num_qubits);

    for (unsigned i = 0; i < num_qubits; ++i) {
      perm.push_back(i);
    }

    unsigned l = 0;
    std::vector<unsigned> qs2(qs.size());

    for (unsigned k = 0; k < qs.size(); ++k) {
      for (unsigned i = 0; i < num_qubits; ++i) {
        if (qs[k] == (unsigned) wire_ordering[i]) {
          qs2[k] = i;
          break;
        }
      }
    }

    for (unsigned k = 0; k < qs2.size(); ++k) {
      if (qs2[k] >= num_local_qubits) {
        unsigned j = 0;
        while (j < qs2.size()) {
          for (j = 0; j < qs2.size(); ++j) {
            if (qs2[j] == l) {
              ++l;

              if (l == num_local_qubits) {
                // We should not get here.
                IO::errorf("error: internal error at %s %d.\n",
                           __FILE__, __LINE__);
                exit(1);
              }

              break;
            }
          }
        }

        std::swap(perm[qs2[k]], perm[l]);
        qs2[k] = l++;
      }
    }

    if (l > 0) {
      ErrorCheck(custatevecExStateVectorPermuteIndexBits(
          state.get(), (int32_t*) perm.data(), num_qubits,
          CUSTATEVEC_EX_PERMUTATION_SCATTER));
    }

    auto f = [&matrix, &state, &num_local_qubits, &qs2](
        unsigned i, const auto& r) {
      void* workspace;
      size_t workspace_size;

      ErrorCheck(custatevecComputeExpectationGetWorkspaceSize(
          r.custatevec_handle, kStateDataType, num_local_qubits, matrix,
          kMatrixDataType, kMatrixLayout, qs2.size(), kComputeType,
          &workspace_size));

      // TODO: reuse allocated memory.
      ErrorCheck(cudaMalloc(&workspace, workspace_size));

      cuDoubleComplex eval;

      ErrorCheck(custatevecComputeExpectation(
          r.custatevec_handle, r.device_ptr, kStateDataType, num_local_qubits,
          &eval, kExpectDataType, nullptr, matrix, kMatrixDataType,
          kMatrixLayout, (int32_t*) qs2.data(), qs2.size(), kComputeType,
          workspace, workspace_size));

      ErrorCheck(cudaFree(workspace));

      return std::complex<double>{cuCreal(eval), cuCimag(eval)};
    };

    return state.reduce(f);
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 32;
  }

 private:
};

}  // namespace qsim

#endif  // SIMULATOR_CUSTATEVECEX_H_
