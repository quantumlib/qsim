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
template <typename For, typename fp_type = float>
class MPSSimulator final {
 public:
  using MPSStateSpace_ = MPSStateSpace<For, fp_type>;
  using State = typename MPSStateSpace_::MPS;

  using Complex = std::complex<fp_type>;
  using Matrix =
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  using OneQBMatrix = Eigen::Matrix<Complex, 2, 2, Eigen::RowMajor>;
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
    const auto bd = state.bond_dim();
    const auto l_offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto r_offset = MPSStateSpace_::GetBlockOffset(state, qs[0] + 1);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQBMap B((Complex*)matrix);
    MatrixMap C((Complex*)(raw_state + end), 2, bd);

    for (unsigned block_sep = l_offset; block_sep < r_offset;
         block_sep += 4 * bd) {
      fp_type* cur_block = raw_state + block_sep;
      ConstMatrixMap A((Complex*)(cur_block), 2, bd);
      C.noalias() = B * A;
      memcpy(cur_block, raw_state + end, sizeof(fp_type) * bd * 4);
    }
  }

  void Apply1Right(const std::vector<unsigned>& qs, const fp_type* matrix,
                   State& state) const {
    fp_type* raw_state = state.get();
    const auto bd = state.bond_dim();
    const auto offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto end = MPSStateSpace_::Size(state);
    ConstOneQBMap B((Complex*)matrix);
    ConstMatrixMap A((Complex*)(raw_state + offset), bd, 2);
    MatrixMap C((Complex*)(raw_state + end), bd, 2);
    C.noalias() = A * B.transpose();
    memcpy(raw_state + offset, raw_state + end, sizeof(fp_type) * bd * 4);
  }

  void ApplyGate2(const std::vector<unsigned>& qs, const fp_type* matrix,
                  State& state) const {
    // TODO: micro-benchmark this function and improve performance.
    const auto bd = state.bond_dim();
    const auto nq = state.num_qubits();
    fp_type* raw_state = state.get();

    const auto i_dim = (qs[0] == 0) ? 1 : bd;
    const auto j_dim = 2;
    const auto k_dim = bd;
    const auto l_dim = 2;
    const auto m_dim = (qs[1] == nq - 1) ? 1 : bd;

    const auto B_0_offset = MPSStateSpace_::GetBlockOffset(state, qs[0]);
    const auto B_1_offset = MPSStateSpace_::GetBlockOffset(state, qs[1]);
    const auto end = MPSStateSpace_::Size(state);

    MatrixMap B_0((Complex*)(raw_state + B_0_offset), i_dim * j_dim, k_dim);
    MatrixMap B_1((Complex*)(raw_state + B_1_offset), k_dim, l_dim * m_dim);

    // Merge both blocks into scratch space.
    MatrixMap C((Complex*)(raw_state + end), i_dim * j_dim, l_dim * m_dim);
    C.noalias() = B_0 * B_1;

    // Transpose inner dims in-place.
    MatrixMap C_t((Complex*)(raw_state + end), i_dim * j_dim * l_dim, m_dim);
    for(unsigned i = 0; i < i_dim * j_dim * l_dim; i += 4){
      C_t.row(i + 1).swap(C_t.row(i + 2));
    }

    // Transpose gate matrix and place in 3rd (last) scratch block.
    const auto scratch3_offset = end + 8 * bd * bd;
    ConstMatrixMap G_mat((Complex*)matrix, 4, 4);
    MatrixMap G_t_mat((Complex*)(raw_state + scratch3_offset), 4, 4);
    G_t_mat = G_mat.transpose();
    G_t_mat.col(1).swap(G_t_mat.col(2));

    // Contract gate and merged block tensors, placing result in B0B1.
    for (unsigned i = 0; i < i_dim; i++) {
      fp_type* src_block = raw_state + end + i * 8 * m_dim;
      fp_type* dest_block = raw_state + B_0_offset + i * 8 * m_dim;
      MatrixMap K_i((Complex*)dest_block, 4, m_dim);
      ConstMatrixMap C_i((Complex*)src_block, 4, m_dim);
      // [i, np, m] = [np, lj] * [i, lj, m]
      K_i.noalias() = G_t_mat * C_i;
    }

    // SVD B0B1.
    MatrixMap K((Complex*)(raw_state + B_0_offset), 2 * i_dim, 2 * m_dim);
    Eigen::BDCSVD<Matrix> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto p = std::min(2 * i_dim, 2 * m_dim);

    // Place U in scratch to truncate and then B0.
    MatrixMap U((Complex*)(raw_state + end), 2 * i_dim, p);
    U.noalias() = svd.matrixU();
    B_0.fill(Complex(0, 0));
    const auto keep_cols = (U.cols() > bd) ? bd : U.cols();
    B_0.block(0, 0, U.rows(), keep_cols).noalias() =
        U(Eigen::all, Eigen::seq(0, keep_cols - 1));

    // Place row product of S V into scratch to truncate and then B1.
    MatrixMap V((Complex*)(raw_state + end), p, 2 * m_dim);
    MatrixMap s_vector((Complex*)(raw_state + end + 8 * bd * bd), p, 1);
    V.noalias() = svd.matrixV().adjoint();
    s_vector.noalias() = svd.singularValues();
    B_1.fill(Complex(0, 0));
    const auto keep_rows = (V.rows() > bd) ? bd : V.rows();
    const auto row_seq = Eigen::seq(0, keep_rows - 1);
    for (unsigned i = 0; i < keep_rows; i++) {
      V.row(i) *= s_vector(i);
    }
    B_1.block(0, 0, keep_rows, V.cols()).noalias() = V(row_seq, Eigen::all);
  }

  For for_;
};

}  // namespace mps
}  // namespace qsim

#endif  // MPS_SIMULATOR_H_
