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

#ifndef MPS_STATESPACE_H_
#define MPS_STATESPACE_H_

// For templates will take care of parallelization.
#define EIGEN_DONT_PARALLELIZE 1

#ifdef _WIN32
#include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

namespace qsim {

namespace mps {

namespace detail {

inline void do_not_free(void*) {}

inline void free(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

}  // namespace detail

/**
 * Class containing context and routines for fixed bond dimension
 * truncated Matrix Product State (MPS) simulation.
 */
template <typename For, typename FP = float>
class MPSStateSpace {
 private:
 public:
  using fp_type = FP;
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;

  using Complex = std::complex<fp_type>;
  using Matrix =
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  // Store MPS tensors with the following shape:
  // [2, bond_dim], [bond_dim, 2, bond_dim], ... , [bond_dim, 2].
  class MPS {
   public:
    MPS() = delete;

    MPS(Pointer&& ptr, unsigned num_qubits, unsigned bond_dim)
        : ptr_(std::move(ptr)), num_qubits_(num_qubits), bond_dim_(bond_dim) {}

    fp_type* get() { return ptr_.get(); }

    const fp_type* get() const { return ptr_.get(); }

    fp_type* release() {
      num_qubits_ = 0;
      return ptr_.release();
    }

    unsigned num_qubits() const { return num_qubits_; }

    unsigned bond_dim() const { return bond_dim_; }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
    unsigned bond_dim_;
  };

  // Note: ForArgs are currently unused.
  template <typename... ForArgs>
  MPSStateSpace(ForArgs&&... args) : for_(args...) {}

  // Requires num_qubits >= 2 and bond_dim >= 2.
  static MPS Create(unsigned num_qubits, unsigned bond_dim) {
    auto end_sizes = 2 * 4 * bond_dim;
    auto internal_sizes = 4 * bond_dim * bond_dim * (num_qubits + 1);
    // Use three extra "internal style" blocks past the end of the
    //   working allocation for scratch space. Needed for gate
    //   application.
    auto size = sizeof(fp_type) * (end_sizes + internal_sizes);

#ifdef _WIN32
    Pointer ptr{(fp_type*)_aligned_malloc(size, 64), &detail::free};
    bool is_null = ptr.get() != nullptr;
    return MPS{std::move(ptr), is_null ? num_qubits : 0,
               is_null ? bond_dim : 0};
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, size) == 0) {
      return MPS{Pointer{(fp_type*)p, &detail::free}, num_qubits, bond_dim};
    } else {
      return MPS{Pointer{nullptr, &detail::free}, 0, 0};
    }
#endif
  }

  static unsigned Size(const MPS& state) {
    auto end_sizes = 2 * 4 * state.bond_dim();
    auto internal_sizes = 4 * state.bond_dim() * state.bond_dim();
    return end_sizes + internal_sizes * (state.num_qubits() - 2);
  }

  static unsigned RawSize(const MPS& state) {
    return sizeof(fp_type) * Size(state);
  }

  // Get the pointer offset to the beginning of an MPS block.
  static unsigned GetBlockOffset(const MPS& state, unsigned i) {
    if (i == 0) {
      return 0;
    }
    return 4 * state.bond_dim() * (1 + state.bond_dim() * (i - 1));
  }

  // Copies the state contents of one MPS to another.
  // Ignores scratch data.
  static bool Copy(const MPS& src, MPS& dest) {
    if ((src.num_qubits() != dest.num_qubits()) ||
        src.bond_dim() != dest.bond_dim()) {
      return false;
    }
    auto size = RawSize(src);
    memcpy(dest.get(), src.get(), size);
    return true;
  }

  // Set the MPS to the |0> state.
  static void SetStateZero(MPS& state) {
    auto size = Size(state);
    memset(state.get(), 0, sizeof(fp_type) * size);
    auto block_size = 4 * state.bond_dim() * state.bond_dim();
    state.get()[0] = 1.0;
    for (unsigned i = 4 * state.bond_dim(); i < size; i += block_size) {
      state.get()[i] = 1.0;
    }
  }

  // Computes Re{<state1 | state2 >} for two equal sized MPS.
  // Requires: state1.bond_dim() == state2.bond_dim() &&
  //           state1.num_qubits() == state2.num_qubits()
  static fp_type RealInnerProduct(MPS& state1, MPS& state2) {
    return InnerProduct(state1, state2).real();
  }

  // Computes <state1 | state2 > for two equal sized MPS.
  // Requires: state1.bond_dim() == state2.bond_dim() &&
  //           state1.num_qubits() == state2.num_qubits()
  static std::complex<fp_type> InnerProduct(MPS& state1, MPS& state2) {
    const auto num_qubits = state1.num_qubits();
    const auto bond_dim = state1.bond_dim();
    const auto end = Size(state1);
    auto offset = 0;
    fp_type* state1_raw = state1.get();
    fp_type* state2_raw = state2.get();

    // Contract leftmost blocks together, store result in state1 scratch.
    ConstMatrixMap top((Complex*)state2_raw, 2, bond_dim);
    ConstMatrixMap bot((Complex*)state1_raw, 2, bond_dim);
    MatrixMap partial_contract((Complex*)(state1_raw + end), bond_dim,
                               bond_dim);
    MatrixMap partial_contract2(
        (Complex*)(state1_raw + end + 2 * bond_dim * bond_dim), bond_dim,
        2 * bond_dim);
    partial_contract.noalias() = top.adjoint() * bot;

    // Contract all internal blocks together.
    for (unsigned i = 1; i < num_qubits - 1; ++i) {
      offset = GetBlockOffset(state1, i);

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state1_raw + end + 2 * bond_dim * bond_dim),
                    bond_dim, 2 * bond_dim);

      // Merge bot into left boundary merged tensor.
      new (&bot) ConstMatrixMap((Complex*)(state1_raw + offset), bond_dim,
                                2 * bond_dim);
      partial_contract2.noalias() = partial_contract * bot;

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state1_raw + end + 2 * bond_dim * bond_dim),
                    2 * bond_dim, bond_dim);

      // Merge top into partial_contract2.
      new (&top) ConstMatrixMap((Complex*)(state2_raw + offset), 2 * bond_dim,
                                bond_dim);
      partial_contract.noalias() = top.adjoint() * partial_contract2;
    }

    // Contract rightmost bottom block.
    offset = GetBlockOffset(state1, num_qubits - 1);
    new (&bot) ConstMatrixMap((Complex*)(state1_raw + offset), bond_dim, 2);
    new (&partial_contract2) MatrixMap(
        (Complex*)(state1_raw + end + 4 * bond_dim * bond_dim), bond_dim, 2);
    partial_contract2.noalias() = partial_contract * bot;

    // Contract rightmost top block.
    new (&top) ConstMatrixMap((Complex*)(state2_raw + offset), 2 * bond_dim, 1);
    new (&partial_contract) MatrixMap((Complex*)(state1_raw + end), 1, 1);
    new (&partial_contract2)
        MatrixMap((Complex*)(state1_raw + end + 4 * bond_dim * bond_dim),
                  2 * bond_dim, 1);
    partial_contract.noalias() = top.adjoint() * partial_contract2;

    return partial_contract(0, 0);
  }

  // Compute the 2x2 1-RDM of state on index. Result written to rdm.
  // Requires: scratch and rdm to be allocated.
  static void ReduceDensityMatrix(MPS& state, MPS& scratch, int index,
                                  fp_type* rdm) {
    const auto num_qubits = state.num_qubits();
    const auto bond_dim = state.bond_dim();
    const auto end = Size(state);
    const bool last_index = (index == num_qubits - 1);
    const auto right_dim = (last_index ? 1 : bond_dim);
    auto offset = 0;
    fp_type* state_raw = state.get();
    fp_type* scratch_raw = scratch.get();
    fp_type* state_raw_workspace = state_raw + end + 2 * bond_dim * bond_dim;
    fp_type* scratch_raw_workspace =
        scratch_raw + end + 2 * bond_dim * bond_dim;

    Copy(state, scratch);

    // Contract leftmost blocks together, store result in state scratch.
    ConstMatrixMap top((Complex*)scratch_raw, 2, bond_dim);
    ConstMatrixMap bot((Complex*)state_raw, 2, bond_dim);
    MatrixMap partial_contract((Complex*)(state_raw + end), bond_dim, bond_dim);
    MatrixMap partial_contract2((Complex*)(state_raw_workspace), bond_dim,
                                2 * bond_dim);

    partial_contract.setZero();
    partial_contract(0, 0) = 1;
    if (index > 0) {
      partial_contract.noalias() = top.adjoint() * bot;
    }

    // Contract all internal blocks together.
    for (unsigned i = 1; i < index; ++i) {
      offset = GetBlockOffset(state, i);

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state_raw_workspace), bond_dim, 2 * bond_dim);

      // Merge bot into left boundary merged tensor.
      new (&bot) ConstMatrixMap((Complex*)(state_raw + offset), bond_dim,
                                2 * bond_dim);
      partial_contract2.noalias() = partial_contract * bot;

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state_raw_workspace), 2 * bond_dim, bond_dim);

      // Merge top into partial_contract2.
      new (&top) ConstMatrixMap((Complex*)(scratch_raw + offset), 2 * bond_dim,
                                bond_dim);
      partial_contract.noalias() = top.adjoint() * partial_contract2;
    }

    // The [bond_dim, bond_dim] block in state_raw now contains the contraction
    // up to, but not including index.
    // Contract rightmost blocks.
    offset = GetBlockOffset(state, num_qubits - 1);
    new (&top) ConstMatrixMap((Complex*)(scratch_raw + offset), bond_dim, 2);
    new (&bot) ConstMatrixMap((Complex*)(state_raw + offset), bond_dim, 2);
    new (&partial_contract)
        MatrixMap((Complex*)(scratch_raw + end), bond_dim, bond_dim);
    new (&partial_contract2)
        MatrixMap((Complex*)(scratch_raw_workspace), bond_dim, 2 * bond_dim);

    partial_contract.setZero();
    partial_contract(0, 0) = 1;
    if (index < num_qubits - 1) {
      partial_contract.noalias() = top * bot.adjoint();
    }

    for (unsigned i = num_qubits - 2; i > index; --i) {
      offset = GetBlockOffset(state, i);

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(scratch_raw_workspace), 2 * bond_dim, bond_dim);

      // Merge bot into left boundary merged tensor.
      new (&bot) ConstMatrixMap((Complex*)(state_raw + offset), 2 * bond_dim,
                                bond_dim);
      partial_contract2.noalias() = bot * partial_contract.adjoint();

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(scratch_raw_workspace), bond_dim, 2 * bond_dim);

      // Merge top into partial_contract2.
      new (&top) ConstMatrixMap((Complex*)(scratch_raw + offset), bond_dim,
                                2 * bond_dim);
      // [bd, bd] = [bd, 2bd] @ [bd, 2bd]
      partial_contract.noalias() = top * partial_contract2.adjoint();
    }

    // The [bond_dim, bond_dim] block in scratch_raw now contains the
    // contraction down from the end, but not including the index. Begin final
    // contraction steps.

    // Get leftmost [bd, bd] contraction and contract with top.

    offset = GetBlockOffset(state, index);
    new (&partial_contract)
        MatrixMap((Complex*)(state_raw + end), bond_dim, bond_dim);
    new (&top)
        ConstMatrixMap((Complex*)(state_raw + offset), bond_dim, 2 * right_dim);
    new (&partial_contract2)
        MatrixMap((Complex*)(scratch_raw_workspace), bond_dim, 2 * right_dim);
    partial_contract2.noalias() = partial_contract * top.conjugate();
    // copy the bottom contraction scratch_raw to state_raw to save space.
    memcpy(state_raw + end, scratch_raw + end,
           bond_dim * bond_dim * 2 * sizeof(fp_type));

    // Contract top again for correct shape.
    fp_type* contract3_target = (last_index ? rdm : scratch_raw);
    MatrixMap partial_contract3((Complex*)contract3_target, 2 * right_dim,
                                2 * right_dim);
    partial_contract3.noalias() = top.transpose() * partial_contract2;

    // If we are contracting the last index, all the needed transforms are done.
    if (last_index) {
      return;
    }

    // Conduct final tensor contraction operations. Cannot be easily compiled to
    // matmul.
    const Eigen::TensorMap<const Eigen::Tensor<Complex, 4, Eigen::RowMajor>>
        t_4d((Complex*)scratch_raw, 2, bond_dim, 2, bond_dim);
    const Eigen::TensorMap<const Eigen::Tensor<Complex, 2, Eigen::RowMajor>>
        t_2d((Complex*)(state_raw + end), bond_dim, bond_dim);

    const Eigen::array<Eigen::IndexPair<int>, 2> product_dims = {
        Eigen::IndexPair<int>(1, 0),
        Eigen::IndexPair<int>(3, 1),
    };
    Eigen::TensorMap<Eigen::Tensor<Complex, 2, Eigen::RowMajor>> out(
        (Complex*)rdm, 2, 2);
    out = t_4d.contract(t_2d, product_dims);
  }

  // Draw a single bitstring sample from state using scratch and scratch2
  // as working space.
  static void SampleOnce(MPS& state, MPS& scratch, MPS& scratch2,
                         std::mt19937* random_gen, std::vector<bool>* sample) {
    // TODO: carefully profile with perf and optimize temp storage
    //  locations for cache friendliness.
    const auto bond_dim = state.bond_dim();
    const auto num_qubits = state.num_qubits();
    const auto end = Size(state);
    const auto left_frontier_offset = GetBlockOffset(state, num_qubits + 1);
    std::default_random_engine generator;
    fp_type* state_raw = state.get();
    fp_type* scratch_raw = scratch.get();
    fp_type* scratch2_raw = scratch2.get();
    fp_type rdm[8];

    sample->reserve(num_qubits);
    Copy(state, scratch);
    Copy(state, scratch2);

    // Store prefix contractions in scratch2.
    auto offset = GetBlockOffset(state, num_qubits - 1);
    ConstMatrixMap top((Complex*)(state_raw + offset), bond_dim, 2);
    ConstMatrixMap bot((Complex*)(scratch_raw + offset), bond_dim, 2);
    MatrixMap partial_contract((Complex*)(scratch2_raw + offset), bond_dim,
                               bond_dim);
    MatrixMap partial_contract2((Complex*)(scratch_raw + end), bond_dim,
                                2 * bond_dim);
    partial_contract.noalias() = top * bot.adjoint();

    for (unsigned i = num_qubits - 2; i > 0; --i) {
      offset = GetBlockOffset(state, i);
      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(scratch_raw + end), 2 * bond_dim, bond_dim);

      // Merge bot into left boundary merged tensor.
      new (&bot) ConstMatrixMap((Complex*)(scratch_raw + offset), 2 * bond_dim,
                                bond_dim);
      partial_contract2.noalias() = bot * partial_contract.adjoint();

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(scratch_raw + end), bond_dim, 2 * bond_dim);

      // Merge top into partial_contract2.
      new (&top) ConstMatrixMap((Complex*)(state_raw + offset), bond_dim,
                                2 * bond_dim);

      // merge into partial_contract -> scracth2_raw.
      new (&partial_contract)
          MatrixMap((Complex*)(scratch2_raw + offset), bond_dim, bond_dim);
      partial_contract.noalias() = top * partial_contract2.adjoint();
    }

    // Compute RDM-0 and draw first sample.
    offset = GetBlockOffset(state, 1);
    new (&top) ConstMatrixMap((Complex*)state_raw, 2, bond_dim);
    new (&bot) ConstMatrixMap((Complex*)scratch_raw, 2, bond_dim);
    new (&partial_contract)
        MatrixMap((Complex*)(scratch2_raw + offset), bond_dim, bond_dim);
    new (&partial_contract2)
        MatrixMap((Complex*)(scratch_raw + end), 2, bond_dim);

    partial_contract2.noalias() = bot * partial_contract.adjoint();

    new (&partial_contract) MatrixMap((Complex*)rdm, 2, 2);
    partial_contract.noalias() = top * partial_contract2.adjoint();
    auto p0 = rdm[0] / (rdm[0] + rdm[6]);
    std::bernoulli_distribution distribution(1 - p0);
    auto bit_val = distribution(*random_gen);
    sample->push_back(bit_val);

    // collapse state.
    new (&partial_contract) MatrixMap((Complex*)scratch_raw, 2, bond_dim);
    partial_contract.row(!bit_val).setZero();

    // Prepare left contraction frontier.
    new (&partial_contract2) MatrixMap(
        (Complex*)(scratch2_raw + left_frontier_offset), bond_dim, bond_dim);
    partial_contract2.noalias() =
        partial_contract.transpose() * partial_contract.conjugate();

    // Compute RDM-i and draw internal tensor samples.
    for (unsigned i = 1; i < num_qubits - 1; i++) {
      // Get leftmost [bd, bd] contraction and contract with top.
      offset = GetBlockOffset(state, i);
      new (&partial_contract) MatrixMap(
          (Complex*)(scratch2_raw + left_frontier_offset), bond_dim, bond_dim);
      new (&top) ConstMatrixMap((Complex*)(state_raw + offset), bond_dim,
                                2 * bond_dim);
      new (&partial_contract2)
          MatrixMap((Complex*)(state_raw + end), bond_dim, 2 * bond_dim);
      partial_contract2.noalias() = partial_contract * top.conjugate();

      // Contract top again for correct shape.
      MatrixMap partial_contract3((Complex*)(scratch_raw + end), 2 * bond_dim,
                                  2 * bond_dim);
      partial_contract3.noalias() = top.transpose() * partial_contract2;

      // Conduct final tensor contraction operations. Cannot be easily compiled
      // to matmul. Perf reports shows only ~6% of runtime spent here on large
      // systems.
      offset = GetBlockOffset(state, i + 1);
      const Eigen::TensorMap<const Eigen::Tensor<Complex, 4, Eigen::RowMajor>>
          t_4d((Complex*)(scratch_raw + end), 2, bond_dim, 2, bond_dim);
      const Eigen::TensorMap<const Eigen::Tensor<Complex, 2, Eigen::RowMajor>>
          t_2d((Complex*)(scratch2_raw + offset), bond_dim, bond_dim);

      const Eigen::array<Eigen::IndexPair<int>, 2> product_dims = {
          Eigen::IndexPair<int>(1, 0),
          Eigen::IndexPair<int>(3, 1),
      };
      Eigen::TensorMap<Eigen::Tensor<Complex, 2, Eigen::RowMajor>> out(
          (Complex*)rdm, 2, 2);
      out = t_4d.contract(t_2d, product_dims);

      // Sample bit and collapse state.
      p0 = rdm[0] / (rdm[0] + rdm[6]);
      distribution = std::bernoulli_distribution(1 - p0);
      bit_val = distribution(*random_gen);

      sample->push_back(bit_val);
      offset = GetBlockOffset(state, i);
      new (&partial_contract)
          MatrixMap((Complex*)(scratch_raw + offset), bond_dim * 2, bond_dim);
      for (unsigned j = !bit_val; j < 2 * bond_dim; j += 2) {
        partial_contract.row(j).setZero();
      }

      // Update left frontier.
      new (&partial_contract) MatrixMap(
          (Complex*)(scratch2_raw + left_frontier_offset), bond_dim, bond_dim);

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state_raw + end), bond_dim, 2 * bond_dim);

      // Merge bot into left boundary merged tensor.
      new (&bot) ConstMatrixMap((Complex*)(scratch_raw + offset), bond_dim,
                                2 * bond_dim);
      partial_contract2.noalias() = partial_contract * bot.conjugate();

      // reshape:
      new (&partial_contract2)
          MatrixMap((Complex*)(state_raw + end), 2 * bond_dim, bond_dim);

      // Merge top into partial_contract2.
      new (&top) ConstMatrixMap((Complex*)(scratch_raw + offset), 2 * bond_dim,
                                bond_dim);
      partial_contract.noalias() = top.transpose() * partial_contract2;
    }

    // Compute RDM-(n-1) and sample.
    offset = GetBlockOffset(state, num_qubits - 1);
    new (&partial_contract2)
        MatrixMap((Complex*)(state_raw + end), bond_dim, 2);

    new (&top) ConstMatrixMap((Complex*)(state_raw + offset), bond_dim, 2);
    partial_contract2.noalias() = partial_contract * top.conjugate();
    new (&partial_contract) MatrixMap((Complex*)rdm, 2, 2);
    partial_contract.noalias() = top.transpose() * partial_contract2;

    p0 = rdm[0] / (rdm[0] + rdm[6]);
    distribution = std::bernoulli_distribution(1 - p0);
    bit_val = distribution(*random_gen);
    sample->push_back(bit_val);
  }

  // Draw num_samples bitstring samples from state and store the result
  // bit vectors in results. Uses scratch and scratch2 as workspace.
  static void Sample(MPS& state, MPS& scratch, MPS& scratch2,
                     unsigned num_samples, unsigned seed,
                     std::vector<std::vector<bool>>* results) {
    std::mt19937 rand_source(seed);
    results->reserve(num_samples);
    for (unsigned i = 0; i < num_samples; i++) {
      SampleOnce(state, scratch, scratch2, &rand_source, &(*results)[i]);
    }
  }

  // Testing only. Convert the MPS to a wavefunction under "normal" ordering.
  // Requires: wf be allocated beforehand with bond_dim * 2 ^ num_qubits -1
  // memory.
  static void ToWaveFunction(MPS& state, fp_type* wf) {
    const auto bond_dim = state.bond_dim();
    const auto num_qubits = state.num_qubits();
    fp_type* raw_state = state.get();

    ConstMatrixMap accum = ConstMatrixMap((Complex*)(raw_state), 2, bond_dim);
    ConstMatrixMap next_block = ConstMatrixMap(nullptr, 0, 0);
    MatrixMap result2 = MatrixMap(nullptr, 0, 0);
    auto offset = 0;
    auto result2_size = 2;

    for (unsigned i = 1; i < num_qubits - 1; i++) {
      offset = GetBlockOffset(state, i);
      // use of new does not trigger any expensive operations.
      new (&next_block) ConstMatrixMap((Complex*)(raw_state + offset), bond_dim,
                                       2 * bond_dim);
      new (&result2) MatrixMap((Complex*)(wf), result2_size, 2 * bond_dim);

      // temp variable used since result2 and accum point to same memory.
      result2 = accum * next_block;
      result2_size *= 2;
      new (&accum) ConstMatrixMap((Complex*)(wf), result2_size, bond_dim);
    }
    offset = GetBlockOffset(state, num_qubits - 1);
    new (&next_block)
        ConstMatrixMap((Complex*)(raw_state + offset), bond_dim, 2);
    new (&result2) MatrixMap((Complex*)(wf), result2_size, 2);
    result2 = accum * next_block;
  }

 protected:
  For for_;
};

}  // namespace mps
}  // namespace qsim

#endif  // MPS_STATESPACE_H_
