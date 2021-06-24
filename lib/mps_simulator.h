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

#ifndef MPS_SIMULATOR_H_
#define MPS_SIMULATOR_H_

// For templates will take care of parallelization.
#define EIGEN_DONT_PARALLELIZE 1

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../eigen/Eigen/Dense"
#include "mps_statespace.h"

namespace qsim {

namespace mps {

/**
 *  Truncated Matrix Product State (MPS) circuit simulator w/ vectorization.
 */
template <typename For, typename fp_type = float>
class MPSSimulator final {
 public:
  using MPSStateSpace_ = MPSStateSpace<For, fp_type>;
  using MPS = typename MPSStateSpace_::MPS;

  using Matrix = Eigen::Matrix<std::complex<fp_type>, Eigen::Dynamic,
                               Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  using OneQBMatrix =
      Eigen::Matrix<std::complex<fp_type>, 2, 2, Eigen::RowMajor>;
  using ConstOneQBMap = Eigen::Map<const OneQBMatrix>;

  // Note: ForArgs are currently unused.
  template <typename... ForArgs>
  explicit MPSSimulator(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix,
                 MPS& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
      case 1:
        ApplyGate1(qs, matrix, state);
        break;
      // case 2:
      //   ApplyGate2(qs, matrix, state);
      //   break;
      // case 3:
      //   ApplyGate3(qs, matrix, state);
      //   break;
      // case 4:
      //   ApplyGate4(qs, matrix, state);
      //   break;
      // case 5:
      //   ApplyGate5(qs, matrix, state);
      //   break;
      // case 6:
      //   ApplyGate6(qs, matrix, state);
      //   break;
      default:
        // Not implemented.
        break;
    }
  }

  /**
   * Applies a controlled gate using eigen3 operations w/ instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, MPS& state) const {
    // TODO.
  }

  /**
   * Computes the expectation value of an operator using eigen3 operations
   * w/ vectorized instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const MPS& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    // TODO.
    return std::complex<double>(-10., -10.);
  }

 private:
  void ApplyGate1(const std::vector<unsigned>& qs, const fp_type* matrix,
                  MPS& state) const {
    if (qs[0] == state.num_qubits() - 1) {
      Apply1Right(qs, matrix, state);
    } else {
      Apply1LeftOrInterior(qs, matrix, state);
    }
  }

  void Apply1LeftOrInterior(const std::vector<unsigned>& qs,
                            const fp_type* matrix, MPS& state) const {
    fp_type* raw_state = state.get();
    const auto bd = state.bond_dim();
    const auto l_offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto r_offset = MPSStateSpace_::GetBlockOffset(state, qs[0] + 1);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQBMap B = ConstOneQBMap((std::complex<fp_type>*)matrix);
    MatrixMap C = MatrixMap((std::complex<fp_type>*)(raw_state + end), 2, bd);

    for (unsigned block_sep = l_offset; block_sep < r_offset;
         block_sep += 4 * bd) {
      fp_type* cur_block = raw_state + block_sep;
      ConstMatrixMap A =
          ConstMatrixMap((std::complex<fp_type>*)(cur_block), 2, bd);
      C.noalias() = B * A;
      memcpy(cur_block, raw_state + end, sizeof(fp_type) * bd * 4);
    }
  }

  void Apply1Right(const std::vector<unsigned>& qs, const fp_type* matrix,
                   MPS& state) const {
    fp_type* raw_state = state.get();
    const auto bd = state.bond_dim();
    const auto offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQBMap B = ConstOneQBMap((std::complex<fp_type>*)matrix);
    ConstMatrixMap A =
        ConstMatrixMap((std::complex<fp_type>*)(raw_state + offset), bd, 2);
    MatrixMap C = MatrixMap((std::complex<fp_type>*)(raw_state + end), bd, 2);
    C.noalias() = A * B.transpose();
    memcpy(raw_state + offset, raw_state + end, sizeof(fp_type) * bd * 4);
  }

  For for_;
};

}  // namespace mps
}  // namespace qsim

#endif  // MPS_SIMULATOR_H_
