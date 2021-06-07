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

namespace qsim {

/**
 * Hybrid Feynman-Schrodinger simulator.
 */
template <typename IO, typename GateT,
          template <typename, typename> class FuserT,
          typename Simulator, typename For>
struct HybridSimulator final {
 public:
  using Gate = GateT;
  using GateKind = typename Gate::GateKind;
  using fp_type = typename Simulator::fp_type;

 private:
  using StateSpace = typename Simulator::StateSpace;
  using State = typename Simulator::State;

  // Note that one can use "struct GateHybrid : public Gate {" in C++17.
  struct GateHybrid {
    using GateKind = HybridSimulator::GateKind;

    GateKind kind;
    unsigned time;
    std::vector<unsigned> qubits;
    std::vector<unsigned> controlled_by;
    uint64_t cmask;
    std::vector<fp_type> params;
    Matrix<fp_type> matrix;
    bool unfusible;
    bool swapped;

    const Gate* parent;
    unsigned id;
  };

  struct GateX {
    GateHybrid* decomposed0;
    GateHybrid* decomposed1;
    schmidt_decomp_type<fp_type> schmidt_decomp;
    unsigned schmidt_bits;
    unsigned swapped;
  };

 public:
  using Fuser = FuserT<IO, GateHybrid>;
  using GateFused = typename Fuser::GateFused;

  /**
   * Contextual data for hybrid simulation.
   */
  struct HybridData {
    /**
     * List of gates on the "0" side of the cut.
     */
    std::vector<GateHybrid> gates0;
    /**
     * List of gates on the "1" side of the cut.
     */
    std::vector<GateHybrid> gates1;
    /**
     * List of gates on the cut.
     */
    std::vector<GateX> gatexs;
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
  struct Parameter : public Fuser::Parameter {
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
   * @param gates List of all gates in the circuit.
   * @param hd Output data with splitted parts.
   * @return True if the splitting done successfully; false otherwise.
   */
  static bool SplitLattice(const std::vector<unsigned>& parts,
                           const std::vector<Gate>& gates, HybridData& hd) {
    hd.num_gatexs = 0;
    hd.num_qubits0 = 0;
    hd.num_qubits1 = 0;

    hd.gates0.reserve(gates.size());
    hd.gates1.reserve(gates.size());
    hd.qubit_map.reserve(parts.size());

    unsigned count0 = 0;
    unsigned count1 = 0;

    // Global qubit index to local qubit index map.
    for (std::size_t i = 0; i < parts.size(); ++i) {
      parts[i] == 0 ? ++hd.num_qubits0 : ++hd.num_qubits1;
      hd.qubit_map.push_back(parts[i] == 0 ? count0++ : count1++);
    }

    // Split the lattice.
    for (const auto& gate : gates) {
      if (gate.kind == gate::kMeasurement) {
        IO::errorf("measurement gates are not suported by qsimh.\n");
        return false;
      }

      if (gate.controlled_by.size() > 0) {
        IO::errorf("controlled gates are not suported by qsimh.\n");
        return false;
      }

      switch (gate.qubits.size()) {
      case 1:  // Single qubit gates.
        switch (parts[gate.qubits[0]]) {
        case 0:
          hd.gates0.emplace_back(GateHybrid{gate.kind, gate.time,
            {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, gate.matrix,
            false, false, nullptr, 0});
          break;
        case 1:
          hd.gates1.emplace_back(GateHybrid{gate.kind, gate.time,
            {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, gate.matrix,
            false, false, nullptr, 0});
          break;
        }
        break;
      case 2:  // Two qubit gates.
        {
          switch ((parts[gate.qubits[1]] << 1) | parts[gate.qubits[0]]) {
          case 0:  // Both qubits in part 0.
            hd.gates0.emplace_back(GateHybrid{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              {}, 0, gate.params, gate.matrix, false, gate.swapped,
              nullptr, 0});
            break;
          case 1:  // Gate on the cut, qubit 0 in part 1, qubit 1 in part 0.
            hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
              {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
              true, gate.swapped, &gate, hd.num_gatexs});
            hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
              true, gate.swapped, &gate, hd.num_gatexs});

            ++hd.num_gatexs;
            break;
          case 2:  // Gate on the cut, qubit 0 in part 0, qubit 1 in part 1.
            hd.gates0.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
              {hd.qubit_map[gate.qubits[0]]}, {}, 0, gate.params, {},
              true, gate.swapped, &gate, hd.num_gatexs});
            hd.gates1.emplace_back(GateHybrid{GateKind::kDecomp, gate.time,
              {hd.qubit_map[gate.qubits[1]]}, {}, 0, gate.params, {},
              true, gate.swapped, &gate, hd.num_gatexs});

            ++hd.num_gatexs;
            break;
          case 3:  // Both qubits in part 1.
            hd.gates1.emplace_back(GateHybrid{gate.kind, gate.time,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              {}, 0, gate.params, gate.matrix, false, gate.swapped,
              nullptr, 0});
            break;
          }
        }
        break;
      default:
        IO::errorf("multi-qubit gates are not suported by qsimh.\n");
        return false;
      }
    }

    auto compare = [](const GateHybrid& l, const GateHybrid& r) -> bool {
      return l.time < r.time || (l.time == r.time &&
          (l.parent < r.parent || (l.parent == r.parent && l.id < r.id)));
    };

    // Sort gates.
    std::sort(hd.gates0.begin(), hd.gates0.end(), compare);
    std::sort(hd.gates1.begin(), hd.gates1.end(), compare);

    hd.gatexs.reserve(hd.num_gatexs);

    // Get Schmidt matrices.
    for (auto& gate0 : hd.gates0) {
      if (gate0.parent != nullptr) {
        auto d = GetSchmidtDecomp(gate0.parent->kind, gate0.parent->params);
        if (d.size() == 0) {
          IO::errorf("no Schmidt decomposition for gate kind %u.\n",
                     gate0.parent->kind);
          return false;
        }

        unsigned schmidt_bits = SchmidtBits(d.size());
        if (schmidt_bits > 2) {
          IO::errorf("Schmidt rank is too large for gate kind %u.\n",
                     gate0.parent->kind);
          return false;
        }

        unsigned swapped = parts[gate0.parent->qubits[0]];
        if (gate0.parent->swapped) swapped = 1 - swapped;
        hd.gatexs.emplace_back(GateX{&gate0, nullptr, std::move(d),
                                     schmidt_bits, swapped});
      }
    }

    unsigned count = 0;
    for (auto& gate1 : hd.gates1) {
      if (gate1.parent != nullptr) {
        hd.gatexs[count++].decomposed1 = &gate1;
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
   * @param hd Container object for gates on the boundary between lattice
   *   sections.
   * @param parts Lattice sections to be simulated.
   * @param fgates0 List of gates from one section of the lattice.
   * @param fgates1 List of gates from the other section of the lattice.
   * @param bitstrings List of output states to simulate, as bitstrings.
   * @param results Output vector of amplitudes. After a successful run, this
   *   will be populated with amplitudes for each state in 'bitstrings'.
   * @return True if the simulation completed successfully; false otherwise.
   */
  bool Run(const Parameter& param, HybridData& hd,
           const std::vector<unsigned>& parts,
           const std::vector<GateFused>& fgates0,
           const std::vector<GateFused>& fgates1,
           const std::vector<uint64_t>& bitstrings,
           std::vector<std::complex<fp_type>>& results) const {
    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    auto bits = CountSchmidtBits(param, hd.gatexs);

    uint64_t rmax = uint64_t{1} << bits.num_r_bits;
    uint64_t smax = uint64_t{1} << bits.num_s_bits;

    auto loc0 = CheckpointLocations(param, fgates0);
    auto loc1 = CheckpointLocations(param, fgates1);

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

    StateSpace sspace(param.num_threads);

    State* rstate0;
    State* rstate1;

    State state0p = sspace.Null();
    State state1p = sspace.Null();
    State state0r = sspace.Null();
    State state1r = sspace.Null();
    State state0s = sspace.Null();
    State state1s = sspace.Null();

    // Create states.

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1,
                      sspace, true, state0p, state1p, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1,
                      sspace, rmax > 1, state0r, state1r, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(hd.num_qubits0, hd.num_qubits1,
                      sspace, smax > 1, state0s, state1s, rstate0, rstate1)) {
      return false;
    }

    sspace.SetStateZero(state0p);
    sspace.SetStateZero(state1p);

    Simulator sim(param.num_threads);

    std::vector<unsigned> prev(hd.num_gatexs, -1);

    // param.prefix encodes the prefix path.
    unsigned gatex_index = SetSchmidtMatrices(
        0, num_p_gates, param.prefix, prev, hd.gatexs);

    if (gatex_index == 0) {
      // Apply gates before the first checkpoint.
      ApplyGates(fgates0, 0, loc0[0], sim, state0p);
      ApplyGates(fgates1, 0, loc1[0], sim, state1p);
    } else {
      IO::errorf("invalid prefix %lu for prefix gate index %u.\n",
                 param.prefix, gatex_index - 1);
      return false;
    }

    // Branch over root gates on the cut. r encodes the root path.
    for (uint64_t r = 0; r < rmax; ++r) {
      if (rmax > 1) {
        sspace.Copy(state0p, state0r);
        sspace.Copy(state1p, state1r);
      }

      if (SetSchmidtMatrices(num_p_gates, num_pr_gates,
                             r, prev, hd.gatexs) == 0) {
        // Apply gates before the second checkpoint.
        ApplyGates(fgates0, loc0[0], loc0[1], sim, state0r);
        ApplyGates(fgates1, loc1[0], loc1[1], sim, state1r);
      } else {
        continue;
      }

      // Branch over suffix gates on the cut. s encodes the suffix path.
      for (uint64_t s = 0; s < smax; ++s) {
        if (smax > 1) {
          sspace.Copy(rmax > 1 ? state0r : state0p, state0s);
          sspace.Copy(rmax > 1 ? state1r : state1p, state1s);
        }

        if (SetSchmidtMatrices(num_pr_gates, hd.num_gatexs,
                               s, prev, hd.gatexs) == 0) {
          // Apply the rest of the gates.
          ApplyGates(fgates0, loc0[1], fgates0.size(), sim, state0s);
          ApplyGates(fgates1, loc1[1], fgates1.size(), sim, state1s);
        } else {
          continue;
        }

        auto f = [](unsigned n, unsigned m, uint64_t i,
                    const StateSpace& sspace,
                    const State& state0, const State& state1,
                    const std::vector<Index>& indices,
                    std::vector<std::complex<fp_type>>& results) {
          auto a0 = sspace.GetAmpl(state0, indices[i].i0);
          auto a1 = sspace.GetAmpl(state1, indices[i].i1);
          results[i] += a0 * a1;
        };

        // Collect results.
        for_.Run(
            results.size(), f, sspace, *rstate0, *rstate1, indices, results);
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
   * @param fgates Set of gates for which to find checkpoint locations.
   * @return A pair of numbers specifying how many gates to apply before the
   *   first and second checkpoints, respectively.
   */
  static std::array<unsigned, 2> CheckpointLocations(
      const Parameter& param, const std::vector<GateFused>& fgates) {
    std::array<unsigned, 2> loc{0, 0};

    unsigned num_decomposed = 0;
    unsigned num_p_gates = param.num_prefix_gatexs;
    unsigned num_pr_gates = num_p_gates + param.num_root_gatexs;

    for (std::size_t i = 0; i < fgates.size(); ++i) {
      for (auto gate: fgates[i].gates) {
        if (gate->parent != nullptr) {
          ++num_decomposed;
          // There should be only one decomposed gate in fused gate.
          break;
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

  static void ApplyGates(const std::vector<GateFused>& gates,
                         std::size_t i0, std::size_t i1,
                         const Simulator& simulator, State& state) {
    for (std::size_t i = i0; i < i1; ++i) {
      ApplyFusedGate(simulator, gates[i], state);
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

  static bool CreateStates(unsigned num_qubits0,unsigned num_qubits1,
                           const StateSpace& sspace,
                           bool create, State& state0, State& state1,
                           State* (&rstate0), State* (&rstate1)) {
    if (create) {
      state0 = sspace.Create(num_qubits0);
      state1 = sspace.Create(num_qubits1);

      if (sspace.IsNull(state0) || sspace.IsNull(state1)) {
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
