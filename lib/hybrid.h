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

#ifndef HYBRID_H_
#define HYBRID_H_

#include <algorithm>
#include <array>
#include <complex>
#include <vector>

#include "gate.h"
#include "gate_appl.h"
#include "gates_cirq.h"
#include "gates_qsim.h"

namespace qsim {

namespace detail {

template <typename fp_type>
inline schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    unsigned kind, const std::vector<fp_type>& params) {
  if (kind >= kGateId1) {
    return qsim::GetSchmidtDecomp(kind, params);
  } else {
    return qsim::Cirq::GetSchmidtDecomp(kind, params);
  }
}

}  // namespace detail

/**
 * Hybrid Feynman-Schrodinger simulator.
 */
template <typename IO, typename For>
struct HybridSimulator final {
 private:
  template <typename FP>
  struct GateX {
    using fp_type = FP;

    DecomposedGate<fp_type>* decomposed0;
    DecomposedGate<fp_type>* decomposed1;
    schmidt_decomp_type<fp_type> schmidt_decomp;
    unsigned schmidt_bits;
    unsigned swapped;
  };

  struct Empty {};

 public:
  /**
   * Contextual data for hybrid simulation.
   */
  template <typename FP>
  struct HybridData {
    using fp_type = FP;
    using OperationD = detail::append_to_variant_t<Operation<fp_type>,
                                                   DecomposedGate<fp_type>>;

    /**
     * List of operations on the "0" side of the cut.
     */
    std::vector<OperationD> ops0;
    /**
     * List of operations on the "1" side of the cut.
     */
    std::vector<OperationD> ops1;
    /**
     * List of gates on the cut.
     */
    std::vector<GateX<fp_type>> gatexs;
    /**
     * Global qubit index to local qubit index map.
     */
    std::vector<unsigned> qubit_map;
    /**
     * Number of qubits on the "0" side of the cut.
     */
    unsigned num_qubits0;
    /**
     * Number of qubits on the "1" side of the cut.
     */
    unsigned num_qubits1;
    /**
     * Number of gates on the cut.
     */
    unsigned num_gatexs;
  };

  /**
   * User-specified parameters for gate fusion and hybrid simulation.
   */
  template <typename ParameterF = Empty>
  struct Parameter : public ParameterF {
    /**
     * Fixed bitstring indicating values to assign to Schmidt decomposition
     * indices of prefix gates.
     */
    uint64_t prefix;
    /**
     * Number of gates on the cut that are part of the prefix. Indices of these
     * gates are assigned the value indicated by `prefix`.
     */
    unsigned num_prefix_gatexs;
    /**
     * Number of gates on the cut that are part of the root. All gates that are
     * not part of the prefix or root are part of the suffix.
     */
    unsigned num_root_gatexs;
    unsigned num_threads;
  };

  template <typename... Args>
  explicit HybridSimulator(Args&&... args) : for_(args...) {}

  /**
   * Splits the lattice into two parts, using Schmidt decomposition for gates
   * on the cut.
   * @param parts Lattice sections to be simulated.
   * @param ops List of all operations in the circuit. Only matrix gates are
   *   supported.
   * @param hd Output data with split parts.
   * @return True if the splitting done successfully; false otherwise.
   */
  template <typename Operation, typename FP>
  static bool SplitLattice(const std::vector<unsigned>& parts,
                           const std::vector<Operation>& ops,
                           HybridData<FP>& hd) {
    using fp_type = FP;
    using Gate = qsim::Gate<fp_type>;
    using DecomposedGate = qsim::DecomposedGate<fp_type>;

    hd.num_gatexs = 0;
    hd.num_qubits0 = 0;
    hd.num_qubits1 = 0;

    hd.ops0.reserve(ops.size());
    hd.ops1.reserve(ops.size());
    hd.qubit_map.reserve(parts.size());

    unsigned count0 = 0;
    unsigned count1 = 0;

    // Global qubit index to local qubit index map.
    for (std::size_t i = 0; i < parts.size(); ++i) {
      parts[i] == 0 ? ++hd.num_qubits0 : ++hd.num_qubits1;
      hd.qubit_map.push_back(parts[i] == 0 ? count0++ : count1++);
    }

    // Split the lattice.
    for (const auto& op : ops) {
      if (!OpGetAlternative<Gate>(op)) {
        IO::errorf("measurement, controlled or other non-matrix gates "
                   "are not suported by qsimh.\n");
        return false;
      }

      const auto& gate = *OpGetAlternative<Gate>(op);

      switch (gate.qubits.size()) {
      case 1:  // Single qubit gates.
        switch (parts[gate.qubits[0]]) {
        case 0:
          hd.ops0.push_back(Gate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, gate.params, gate.matrix});
          break;
        case 1:
          hd.ops1.push_back(Gate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, gate.params, gate.matrix});
          break;
        }
        break;
      case 2:  // Two qubit gates.
        switch ((parts[gate.qubits[1]] << 1) | parts[gate.qubits[0]]) {
        case 0:  // Both qubits in part 0.
          hd.ops0.push_back(Gate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              gate.params, gate.matrix});
          break;
        case 1:  // Gate on the cut, qubit 0 in part 1, qubit 1 in part 0.
          hd.ops0.push_back(DecomposedGate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[1]]}, gate.params, {},
              gate.swapped, &gate, hd.num_gatexs});
          hd.ops1.push_back(DecomposedGate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, gate.params, {},
              gate.swapped, &gate, hd.num_gatexs});

          ++hd.num_gatexs;
          break;
        case 2:  // Gate on the cut, qubit 0 in part 0, qubit 1 in part 1.
          hd.ops0.push_back(DecomposedGate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, gate.params, {},
              gate.swapped, &gate, hd.num_gatexs});
          hd.ops1.push_back(DecomposedGate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[1]]}, gate.params, {},
              gate.swapped, &gate, hd.num_gatexs});

          ++hd.num_gatexs;
          break;
        case 3:  // Both qubits in part 1.
          hd.ops1.push_back(Gate{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              gate.params, gate.matrix});
          break;
        }
        break;
      default:
        IO::errorf("multi-qubit gates are not suported by qsimh.\n");
        return false;
      }
    }

    using OperationD = typename HybridData<fp_type>::OperationD;

    auto compare = [](const OperationD& lop, const OperationD& rop) -> bool {
      unsigned ltime = OpTime(lop);
      unsigned rtime = OpTime(rop);

      const auto* ld = OpGetAlternative<DecomposedGate>(lop);
      const auto* rd = OpGetAlternative<DecomposedGate>(rop);

      return ltime < rtime ||
          (ltime == rtime && ((!ld && rd) || (ld && rd && ld->id < rd->id)));
    };

    // Sort ops.
    std::sort(hd.ops0.begin(), hd.ops0.end(), compare);
    std::sort(hd.ops1.begin(), hd.ops1.end(), compare);

    hd.gatexs.reserve(hd.num_gatexs);

    // Get Schmidt matrices.
    for (auto& op0 : hd.ops0) {
      if (auto* pg = OpGetAlternative<DecomposedGate>(op0)) {
        auto d = detail::GetSchmidtDecomp(pg->parent->kind, pg->parent->params);
        if (d.size() == 0) {
          IO::errorf("no Schmidt decomposition for gate kind %u.\n",
                     pg->parent->kind);
          return false;
        }

        unsigned schmidt_bits = SchmidtBits(d.size());
        if (schmidt_bits > 2) {
          IO::errorf("Schmidt rank is too large for gate kind %u.\n",
                     pg->parent->kind);
          return false;
        }

        unsigned swapped = parts[pg->parent->qubits[0]];
        if (pg->parent->swapped) swapped = 1 - swapped;

        hd.gatexs.push_back(GateX<fp_type>{pg, nullptr, std::move(d),
                                           schmidt_bits, swapped});
      }
    }

    unsigned count = 0;
    for (auto& op1 : hd.ops1) {
      if (auto* pg = OpGetAlternative<DecomposedGate>(op1)) {
        hd.gatexs[count++].decomposed1 = pg;
      }
    }

    for (auto& gatex : hd.gatexs) {
      if (gatex.schmidt_decomp.size() == 1) {
        FillSchmidtMatrices(0, gatex);
      }
    }

    return true;
  }

  /**
   * Runs the hybrid simulator on a sectioned lattice.
   * @param param Options for parallelism and logging. Also specifies the size
   *   of the 'prefix' and 'root' sections of the lattice.
   * @param factory Object to create simulators and state spaces.
   * @param hd Container object for gates on the boundary between lattice
   *   sections.
   * @param parts Lattice sections to be simulated.
   * @param ops0 List of (fused) operations from one section of the lattice.
   * @param ops1 List of (fused) operations from the other section of
   *   the lattice.
   * @param bitstrings List of output states to simulate, as bitstrings.
   * @param results Output vector of amplitudes. After a successful run, this
   *   will be populated with amplitudes for each state in 'bitstrings'.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename ParameterF, typename Factory, typename FP,
            typename OperationF, typename Results>
  bool Run(const Parameter<ParameterF>& param, const Factory& factory,
           HybridData<FP>& hd, const std::vector<unsigned>& parts,
           const std::vector<OperationF>& ops0,
           const std::vector<OperationF>& ops1,
           const std::vector<uint64_t>& bitstrings, Results& results) const {
    using Simulator = typename Factory::Simulator;
    using StateSpace = typename Simulator::StateSpace;
    using State = typename StateSpace::State;

    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    auto bits = CountSchmidtBits(param, hd.gatexs);

    uint64_t rmax = uint64_t{1} << bits.num_r_bits;
    uint64_t smax = uint64_t{1} << bits.num_s_bits;

    auto loc0 = CheckpointLocations(param, ops0);
    auto loc1 = CheckpointLocations(param, ops1);

    struct Index {
      unsigned i0;
      unsigned i1;
    };

    std::vector<Index> indices;
    indices.reserve(bitstrings.size());

    // Bitstring indices for part 0 and part 1. TODO: optimize.
    for (const auto& bitstring : bitstrings) {
      Index index{0, 0};

      for (uint64_t i = 0; i < hd.qubit_map.size(); ++i) {
        unsigned m = ((bitstring >> i) & 1) << hd.qubit_map[i];
        parts[i] ? index.i1 |= m : index.i0 |= m;
      }

      indices.push_back(index);
    }

    StateSpace state_space = factory.CreateStateSpace();

    State* rstate0;
    State* rstate1;

    State state0p = state_space.Null();
    State state1p = state_space.Null();
    State state0r = state_space.Null();
    State state1r = state_space.Null();
    State state0s = state_space.Null();
    State state1s = state_space.Null();

    // Create states.

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, state_space, true,
                      state0p, state1p, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, state_space, rmax > 1,
                      state0r, state1r, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1, state_space, smax > 1,
                      state0s, state1s, rstate0, rstate1)) {
      return false;
    }

    state_space.SetStateZero(state0p);
    state_space.SetStateZero(state1p);

    Simulator simulator = factory.CreateSimulator();

    std::vector<unsigned> prev(hd.num_gatexs, unsigned(-1));

    // param.prefix encodes the prefix path.
    unsigned gatex_index = SetSchmidtMatrices(
        0, num_p_gates, param.prefix, prev, hd.gatexs);

    if (gatex_index == 0) {
      // Apply gates before the first checkpoint.
      ApplyGates(ops0, 0, loc0[0], simulator, state0p);
      ApplyGates(ops1, 0, loc1[0], simulator, state1p);
    } else {
      IO::errorf("invalid prefix %lu for prefix gate index %u.\n",
                 param.prefix, gatex_index - 1);
      return false;
    }

    // Branch over root gates on the cut. r encodes the root path.
    for (uint64_t r = 0; r < rmax; ++r) {
      if (rmax > 1) {
        state_space.Copy(state0p, state0r);
        state_space.Copy(state1p, state1r);
      }

      if (SetSchmidtMatrices(num_p_gates, num_pr_gates,
                             r, prev, hd.gatexs) == 0) {
        // Apply gates before the second checkpoint.
        ApplyGates(ops0, loc0[0], loc0[1], simulator, state0r);
        ApplyGates(ops1, loc1[0], loc1[1], simulator, state1r);
      } else {
        continue;
      }

      // Branch over suffix gates on the cut. s encodes the suffix path.
      for (uint64_t s = 0; s < smax; ++s) {
        if (smax > 1) {
          state_space.Copy(rmax > 1 ? state0r : state0p, state0s);
          state_space.Copy(rmax > 1 ? state1r : state1p, state1s);
        }

        if (SetSchmidtMatrices(num_pr_gates, hd.num_gatexs,
                               s, prev, hd.gatexs) == 0) {
          // Apply the rest of the gates.
          ApplyGates(ops0, loc0[1], ops0.size(), simulator, state0s);
          ApplyGates(ops1, loc1[1], ops1.size(), simulator, state1s);
        } else {
          continue;
        }

        auto f = [](unsigned n, unsigned m, uint64_t i,
                    const StateSpace& state_space,
                    const State& state0, const State& state1,
                    const std::vector<Index>& indices, Results& results) {
          // TODO: make it faster for the CUDA state space.
          auto a0 = state_space.GetAmpl(state0, indices[i].i0);
          auto a1 = state_space.GetAmpl(state1, indices[i].i1);
          results[i] += a0 * a1;
        };

        // Collect results.
        for_.Run(results.size(), f,
                 state_space, *rstate0, *rstate1, indices, results);
      }
    }
    return true;
  }

 private:
  /**
   * Identifies when to save "checkpoints" of the simulation state. These allow
   * runs with different cut-index values to reuse parts of the simulation.
   * @param param Options for parallelism and logging. Also specifies the size
   *   of the 'prefix' and 'root' sections of the lattice.
   * @param ops Set of (fused) operations for which to find checkpoint
   *   locations.
   * @return A pair of numbers specifying how many gates to apply before the
   *   first and second checkpoints, respectively.
   */
  template <typename Parameter, typename OperationF>
  static std::array<unsigned, 2> CheckpointLocations(
      const Parameter& param, const std::vector<OperationF>& ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;

    std::array<unsigned, 2> loc{0, 0};

    unsigned num_decomposed = 0;
    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    for (std::size_t i = 0; i < ops.size(); ++i) {
      if (const auto* pg = OpGetAlternative<FusedGate>(ops[i])) {
        if (pg->ParentIsDecomposed()) {
          ++num_decomposed;
        }
      }

      if (num_decomposed <= num_p_gates) {
        loc[0] = i + 1;
      }

      if (num_decomposed <= num_pr_gates) {
        loc[1] = i + 1;
      }
    }

    return loc;
  }

  struct Bits {
    unsigned num_p_bits;
    unsigned num_r_bits;
    unsigned num_s_bits;
  };

  template <typename Parameter, typename GateX>
  static Bits CountSchmidtBits(
      const Parameter& param, const std::vector<GateX>& gatexs) {
    Bits bits{0, 0, 0};

    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    for (std::size_t i = 0; i < gatexs.size(); ++i) {
      const auto& gatex = gatexs[i];
      if (i < num_p_gates) {
        bits.num_p_bits += gatex.schmidt_bits;
      } else if (i < num_pr_gates) {
        bits.num_r_bits += gatex.schmidt_bits;
      } else {
        bits.num_s_bits += gatex.schmidt_bits;
      }
    }

    return bits;
  }

  template <typename GateX>
  static unsigned SetSchmidtMatrices(std::size_t i0, std::size_t i1,
                                     uint64_t path,
                                     std::vector<unsigned>& prev_k,
                                     std::vector<GateX>& gatexs) {
    unsigned shift_length = 0;

    for (std::size_t i = i0; i < i1; ++i) {
      const auto& gatex = gatexs[i];

      if (gatex.schmidt_bits == 0) {
        // Continue if gatex has Schmidt rank 1.
        continue;
      }

      unsigned k = (path >> shift_length) & ((1 << gatex.schmidt_bits) - 1);
      shift_length += gatex.schmidt_bits;

      if (k != prev_k[i]) {
        if (k >= gatex.schmidt_decomp.size()) {
          // Invalid path. Returns gatex index plus one to report error in case
          // of invalid prefix.
          return i + 1;
        }

        FillSchmidtMatrices(k, gatex);

        prev_k[i] = k;
      }
    }

    return 0;
  }

  template <typename GateX>
  static void FillSchmidtMatrices(unsigned k, const GateX& gatex) {
    unsigned part0 = gatex.swapped;
    unsigned part1 = 1 - part0;
    {
      gatex.decomposed0->matrix.resize(gatex.schmidt_decomp[k][part0].size());
      auto begin = gatex.schmidt_decomp[k][part0].begin();
      auto end = gatex.schmidt_decomp[k][part0].end();
      std::copy(begin, end, gatex.decomposed0->matrix.begin());
    }
    {
      gatex.decomposed1->matrix.resize(gatex.schmidt_decomp[k][part1].size());
      auto begin = gatex.schmidt_decomp[k][part1].begin();
      auto end = gatex.schmidt_decomp[k][part1].end();
      std::copy(begin, end, gatex.decomposed1->matrix.begin());
    }
  }

  template <typename OperationF, typename Simulator>
  static void ApplyGates(const std::vector<OperationF>& ops,
                         std::size_t i0, std::size_t i1,
                         const Simulator& simulator,
                         typename Simulator::State& state) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;

    for (std::size_t i = i0; i < i1; ++i) {
      if (const auto* pg = OpGetAlternative<FusedGate>(ops[i])) {
        if (!pg->ParentIsDecomposed()) {
          ApplyGate(simulator, ops[i], state);
        } else {
          auto fgate = *pg;
          CalculateFusedMatrix(fgate);
          ApplyGate(simulator, fgate, state);
        }
      } else {
        ApplyGate(simulator, ops[i], state);
      }
    }
  }

  static unsigned SchmidtBits(unsigned size) {
    switch (size) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 3:
      return 2;
    case 4:
      return 2;
    default:
      // Not supported.
      return 42;
    }
  }

  template <typename StateSpace>
  static bool CreateStates(unsigned num_qubits0,unsigned num_qubits1,
                           const StateSpace& state_space, bool create,
                           typename StateSpace::State& state0,
                           typename StateSpace::State& state1,
                           typename StateSpace::State* (&rstate0),
                           typename StateSpace::State* (&rstate1)) {
    if (create) {
      state0 = state_space.Create(num_qubits0);
      state1 = state_space.Create(num_qubits1);

      if (state_space.IsNull(state0) || state_space.IsNull(state1)) {
        IO::errorf("not enough memory: is the number of qubits too large?\n");
        return false;
      }

      rstate0 = &state0;
      rstate1 = &state1;
    }

    return true;
  }

  For for_;
};

}  // namespace qsim

#endif  // HYBRID_H_
