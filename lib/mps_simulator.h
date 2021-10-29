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
#include "../eigen/Eigen/SVD"
#include "mps_statespace.h"

namespace qsim {

namespace mps {

/**
 *  Truncated Matrix Product State (MPS) circuit simulator w/ vectorization.
 */
template <typename For, typename FP = float>
class MPSSimulator final {
 public:
  using MPSStateSpace_ = MPSStateSpace<For, FP>;
  using State = typename MPSStateSpace_::MPS;
  using fp_type = typename MPSStateSpace_::fp_type;

  using Complex = std::complex<fp_type>;
  using Matrix =
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  using OneQubitMatrix = Eigen::Matrix<Complex, 2, 2, Eigen::RowMajor>;
  using ConstOneQubitMap = Eigen::Map<const OneQubitMatrix>;

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
                 State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
      case 1:
        ApplyGate1(qs, matrix, state);
        break;
      case 2:
        ApplyGate2(qs, matrix, state);
        break;
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
                           const fp_type* matrix, State& state) const {
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
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    // TODO.
    return std::complex<double>(-10., -10.);
  }

 private:
  void ApplyGate1(const std::vector<unsigned>& qs, const fp_type* matrix,
                  State& state) const {
    if (qs[0] == state.num_qubits() - 1) {
      Apply1Right(qs, matrix, state);
    } else {
      Apply1LeftOrInterior(qs, matrix, state);
    }
  }

  void Apply1LeftOrInterior(const std::vector<unsigned>& qs,
                            const fp_type* matrix, State& state) const {
    fp_type* raw_state = state.get();
    const auto bond_dim = state.bond_dim();
    const auto l_offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto r_offset = MPSStateSpace_::GetBlockOffset(state, qs[0] + 1);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQubitMap gate_matrix((Complex*) matrix);
    MatrixMap scratch_block((Complex*)(raw_state + end), 2, bond_dim);

    for (unsigned block_sep = l_offset; block_sep < r_offset;
         block_sep += 4 * bond_dim) {
      fp_type* cur_block = raw_state + block_sep;
      ConstMatrixMap mps_block((Complex*) cur_block, 2, bond_dim);
      scratch_block.noalias() = gate_matrix * mps_block;
      memcpy(cur_block, raw_state + end, sizeof(fp_type) * bond_dim * 4);
    }
  }

  void Apply1Right(const std::vector<unsigned>& qs, const fp_type* matrix,
                   State& state) const {
    fp_type* raw_state = state.get();
    const auto bond_dim = state.bond_dim();
    const auto offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQubitMap gate_matrix((Complex*) matrix);
    ConstMatrixMap mps_block((Complex*)(raw_state + offset), bond_dim, 2);
    MatrixMap scratch_block((Complex*)(raw_state + end), bond_dim, 2);
    scratch_block.noalias() = mps_block * gate_matrix.transpose();
    memcpy(raw_state + offset, raw_state + end, sizeof(fp_type) * bond_dim * 4);
  }

  void ApplyGate2(const std::vector<unsigned>& qs, const fp_type* matrix,
                  State& state) const {
    // TODO: micro-benchmark this function and improve performance.
    const auto bond_dim = state.bond_dim();
    const auto num_qubits = state.num_qubits();
    fp_type* raw_state = state.get();

    const auto i_dim = (qs[0] == 0) ? 1 : bond_dim;
    const auto j_dim = 2;
    const auto k_dim = bond_dim;
    const auto l_dim = 2;
    const auto m_dim = (qs[1] == num_qubits - 1) ? 1 : bond_dim;

    const auto b_0_offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto b_1_offset = MPSStateSpace_::GetBlockOffset(state, qs[1]);
    const auto end = MPSStateSpace_::Size(state);

    MatrixMap block_0((Complex*)(raw_state + b_0_offset), i_dim * j_dim, k_dim);
    MatrixMap block_1((Complex*)(raw_state + b_1_offset), k_dim, l_dim * m_dim);

    // Merge both blocks into scratch space.
    MatrixMap scratch_c((Complex*)(raw_state + end), i_dim * j_dim, l_dim * m_dim);
    scratch_c.noalias() = block_0 * block_1;

    // Transpose inner dims in-place.
    MatrixMap scratch_c_t((Complex*)(raw_state + end), i_dim * j_dim * l_dim, m_dim);
    for (unsigned i = 0; i < i_dim * j_dim * l_dim; i += 4) {
      scratch_c_t.row(i + 1).swap(scratch_c_t.row(i + 2));
    }

    // Transpose gate matrix and place in 3rd (last) scratch block.
    const auto scratch3_offset = end + 8 * bond_dim * bond_dim;
    ConstMatrixMap gate_matrix((Complex*) matrix, 4, 4);
    MatrixMap gate_matrix_transpose((Complex*)(raw_state + scratch3_offset), 4, 4);
    gate_matrix_transpose = gate_matrix.transpose();
    gate_matrix_transpose.col(1).swap(gate_matrix_transpose.col(2));

    // Contract gate and merged block tensors, placing result in B0B1.
    for (unsigned i = 0; i < i_dim; ++i) {
      fp_type* src_block = raw_state + end + i * 8 * m_dim;
      fp_type* dest_block = raw_state + b_0_offset + i * 8 * m_dim;
      MatrixMap block_b0b1((Complex*) dest_block, 4, m_dim);
      ConstMatrixMap scratch_c_i((Complex*) src_block, 4, m_dim);
      // [i, np, m] = [np, lj] * [i, lj, m]
      block_b0b1.noalias() = gate_matrix_transpose * scratch_c_i;
    }

    // SVD B0B1.
    MatrixMap full_b0b1((Complex*)(raw_state + b_0_offset), 2 * i_dim, 2 * m_dim);
    Eigen::BDCSVD<Matrix> svd(full_b0b1, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto p = std::min(2 * i_dim, 2 * m_dim);

    // Place U in scratch to truncate and then B0.
    MatrixMap svd_u((Complex*)(raw_state + end), 2 * i_dim, p);
    svd_u.noalias() = svd.matrixU();
    block_0.fill(Complex(0, 0));
    const auto keep_cols = (svd_u.cols() > bond_dim) ? bond_dim : svd_u.cols();
    block_0.block(0, 0, svd_u.rows(), keep_cols).noalias() =
        svd_u(Eigen::indexing::all, Eigen::seq(0, keep_cols - 1));

    // Place row product of S V into scratch to truncate and then B1.
    MatrixMap svd_v((Complex*)(raw_state + end), p, 2 * m_dim);
    MatrixMap s_vector((Complex*)(raw_state + end + 8 * bond_dim * bond_dim), p, 1);
    svd_v.noalias() = svd.matrixV().adjoint();
    s_vector.noalias() = svd.singularValues();
    block_1.fill(Complex(0, 0));
    const auto keep_rows = (svd_v.rows() > bond_dim) ? bond_dim : svd_v.rows();
    const auto row_seq = Eigen::seq(0, keep_rows - 1);
    for (unsigned i = 0; i < keep_rows; ++i) {
      svd_v.row(i) *= s_vector(i);
    }
    block_1.block(0, 0, keep_rows, svd_v.cols()).noalias() =
        svd_v(row_seq, Eigen::indexing::all);
  }

  For for_;
};

}  // namespace mps
}  // namespace qsim

#endif  // MPS_SIMULATOR_H_
