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
#include <vector>

#include "fuser_basic.h"
#include "gates_appl.h"
#include "gates_def.h"
#include "util.h"

namespace qsim {

// Hybrid Feynmann-Schrodiner simulator.
template <typename IO, template <typename> class FuserT, typename Simulator,
          typename ParallelFor>
struct HybridSimulator final {
 public:
  using fp_type = typename Simulator::fp_type;

 private:
  using StateSpace = typename Simulator::StateSpace;
  using State = typename Simulator::State;
  using Gate = qsim::Gate<fp_type>;

  // Note that one can use "struct GateHybrid : public Gate {" in C++17.
  struct GateHybrid {
    GateKind kind;
    unsigned time;
    unsigned num_qubits;
    unsigned qubits[3];
    bool unfusible;
    bool inverse;
    std::vector<fp_type> params;
    std::array<fp_type, 32> matrix;
    const Gate* parent;
    unsigned id;
  };

  struct GateX {
    GateHybrid* decomposed0;
    GateHybrid* decomposed1;
    schmidt_decomp_type<fp_type> schmidt_decomp;
    unsigned schmidt_bits;
    unsigned inverse;
  };

 public:
  using Fuser = FuserT<GateHybrid>;
  using GateFused = typename Fuser::GateFused;

  struct HybridData {
    std::vector<GateHybrid> gates0;
    std::vector<GateHybrid> gates1;
    std::vector<GateX> gatexs;
    std::vector<unsigned> qubit_map;
    unsigned num_qubits0;
    unsigned num_qubits1;
    unsigned num_gatexs;  // Number of gates of the cut.
  };

  struct Parameter {
    uint64_t prefix;
    unsigned num_prefix_gatexs;
    unsigned num_root_gatexs;
    unsigned num_threads;
    unsigned verbosity = 0;
  };

  // Split the lattice into two parts.
  // Use Schmidt decomposition for gates on the cut.
  static HybridData SplitLattice(const std::vector<unsigned>& parts,
                                 const std::vector<Gate>& gates) {
    HybridData hd;

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
      switch (gate.num_qubits) {
      case 1:  // Single qubit gates.
        switch (parts[gate.qubits[0]]) {
        case 0:
          hd.gates0.push_back({gate.kind, gate.time, 1,
            {hd.qubit_map[gate.qubits[0]]}, false, false, gate.params,
             gate.matrix, nullptr, 0});
          break;
        case 1:
          hd.gates1.push_back(GateHybrid{gate.kind, gate.time, 1,
            {hd.qubit_map[gate.qubits[0]]}, false, false, gate.params,
             gate.matrix, nullptr});
          break;
        }
        break;
      case 2:  // Two qubit gates.
        {
          switch ((parts[gate.qubits[1]] << 1) | parts[gate.qubits[0]]) {
          case 0:  // Both qubits in part 0.
            hd.gates0.push_back(GateHybrid{gate.kind, gate.time, 2,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              false, gate.inverse, gate.params, gate.matrix, nullptr});
            break;
          case 1:  // Gate on the cut, qubit 0 in part 1, qubit 1 in part 0.
            hd.gates0.push_back(GateHybrid{kGateDecomp, gate.time, 1,
              {hd.qubit_map[gate.qubits[1]]}, true, gate.inverse, gate.params,
              {}, &gate, hd.num_gatexs});
            hd.gates1.push_back(GateHybrid{kGateDecomp, gate.time, 1,
              {hd.qubit_map[gate.qubits[0]]}, true, gate.inverse, gate.params,
              {}, &gate, hd.num_gatexs});

            ++hd.num_gatexs;
            break;
          case 2:  // Gate on the cut, qubit 0 in part 0, qubit 1 in part 1.
            hd.gates0.push_back(GateHybrid{kGateDecomp, gate.time, 1,
              {hd.qubit_map[gate.qubits[0]]}, true, gate.inverse, gate.params,
              {}, &gate, hd.num_gatexs});
            hd.gates1.push_back(GateHybrid{kGateDecomp, gate.time, 1,
              {hd.qubit_map[gate.qubits[1]]}, true, gate.inverse, gate.params,
              {}, &gate, hd.num_gatexs});

            ++hd.num_gatexs;
            break;
          case 3:  // Both qubits in part 1.
            hd.gates1.push_back(GateHybrid{gate.kind, gate.time, 2,
              {hd.qubit_map[gate.qubits[0]], hd.qubit_map[gate.qubits[1]]},
              false, gate.inverse, gate.params, gate.matrix, nullptr});
            break;
          }
        }
        break;
      default:
        // Not supported.
        break;
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
        unsigned schmidt_bits = SchmidtBits(d.size());
        unsigned inverse = parts[gate0.parent->qubits[0]];
        if (gate0.parent->inverse) inverse = 1 - inverse;
        hd.gatexs.push_back(GateX{&gate0, nullptr, std::move(d), schmidt_bits,
                                  inverse});
      }
    }

    unsigned count = 0;
    for (auto& gate1 : hd.gates1) {
      if (gate1.parent != nullptr) {
        hd.gatexs[count++].decomposed1 = &gate1;
      }
    }

    return hd;
  }

  // Run hybrid simulator.
  static bool Run(const Parameter& param, HybridData& hd,
                  const std::vector<unsigned>& parts,
                  const std::vector<GateFused>& fgates0,
                  const std::vector<GateFused>& fgates1,
                  const std::vector<uint64_t>& bitstrings,
                  std::vector<std::complex<fp_type>>& results) {
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

    StateSpace sspace0(hd.num_qubits0, param.num_threads);
    StateSpace sspace1(hd.num_qubits1, param.num_threads);

    State* rstate0;
    State* rstate1;

    State state0p = sspace0.NullState();
    State state1p = sspace1.NullState();
    State state0r = sspace0.NullState();
    State state1r = sspace1.NullState();
    State state0s = sspace0.NullState();
    State state1s = sspace1.NullState();

    // Create states.

    if (!CreateStates(
        sspace0, sspace1, true, state0p, state1p, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(
        sspace0, sspace1, rmax > 1, state0r, state1r, rstate0, rstate1)) {
      return false;
    }

    if (!CreateStates(
        sspace0, sspace1, smax > 1, state0s, state1s, rstate0, rstate1)) {
      return false;
    }

    sspace0.SetStateZero(state0p);
    sspace1.SetStateZero(state1p);

    Simulator sim0(hd.num_qubits0, param.num_threads);
    Simulator sim1(hd.num_qubits1, param.num_threads);

    std::vector<unsigned> prev(hd.num_gatexs, -1);

    SetSchmidtMatrices(0, num_p_gates, param.prefix, prev, hd.gatexs);

    // Apply gates before the first checkpoint.
    ApplyGates(fgates0, 0, loc0[0], sim0, state0p);
    ApplyGates(fgates1, 0, loc1[0], sim1, state1p);

    // Branch over root gates on the cut.
    for (uint64_t r = 0; r < rmax; ++r) {
      if (rmax > 1) {
        sspace0.CopyState(state0p, state0r);
        sspace1.CopyState(state1p, state1r);
      }

      SetSchmidtMatrices(num_p_gates, num_pr_gates, r, prev, hd.gatexs);

      // Apply gates before the second checkpoint.
      ApplyGates(fgates0, loc0[0], loc0[1], sim0, state0r);
      ApplyGates(fgates1, loc1[0], loc1[1], sim1, state1r);

      // Branch over suffix gates on the cut.
      for (uint64_t s = 0; s < smax; ++s) {
        if (smax > 1) {
          sspace0.CopyState(rmax > 1 ? state0r : state0p, state0s);
          sspace1.CopyState(rmax > 1 ? state1r : state1p, state1s);
        }

        SetSchmidtMatrices(num_pr_gates, hd.num_gatexs, s, prev, hd.gatexs);

        // Apply the rest of the gates.
        ApplyGates(fgates0, loc0[1], fgates0.size(), sim0, state0s);
        ApplyGates(fgates1, loc1[1], fgates1.size(), sim1, state1s);

        auto f = [](unsigned n, unsigned m, uint64_t i,
                    const StateSpace& sspace0, const StateSpace& sspace1,
                    const State& state0, const State& state1,
                    const std::vector<Index>& indices,
                    std::vector<std::complex<fp_type>>& results) {
          auto a0 = sspace0.GetAmpl(state0, indices[i].i0);
          auto a1 = sspace1.GetAmpl(state1, indices[i].i1);
          results[i] += a0 * a1;
        };

        // Collect results.
        ParallelFor::Run(param.num_threads, results.size(), f,
                         sspace0, sspace1, *rstate0, *rstate1, indices,
                         results);
      }
    }

    return true;
  }

 private:
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

  static void SetSchmidtMatrices(std::size_t i0, std::size_t i1,
                                 uint64_t mask, std::vector<unsigned>& prev_k,
                                 std::vector<GateX>& gatexs) {
    unsigned shift_length = 0;

    for (std::size_t i = i0; i < i1; ++i) {
      const auto& gatex = gatexs[i];
      unsigned k = (mask >> shift_length) & ((1 << gatex.schmidt_bits) - 1);
      shift_length += gatex.schmidt_bits;
      if (k != prev_k[i]) {
        unsigned part0 = gatex.inverse;
        unsigned part1 = 1 - part0;
        {
          auto begin = gatex.schmidt_decomp[k][part0].begin();
          auto end = gatex.schmidt_decomp[k][part0].end();
          std::copy(begin, end, gatex.decomposed0->matrix.begin());
        }
        {
          auto begin = gatex.schmidt_decomp[k][part1].begin();
          auto end = gatex.schmidt_decomp[k][part1].end();
          std::copy(begin, end, gatex.decomposed1->matrix.begin());
        }
        prev_k[i] = k;
      }
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
    if (size >= 1 && size <= 2) {
      return 1;
    } else if (size >= 3 && size <= 4) {
      return 2;
    }

    // Not supported.
    return 0;
  }

  static bool CreateStates(
      const StateSpace& sspace0, const StateSpace& sspace1,
      bool create, State& state0, State& state1,
      State* (&rstate0), State* (&rstate1)) {
    if (create) {
      state0 = sspace0.CreateState();
      state1 = sspace1.CreateState();

      if (sspace0.IsNull(state0) || sspace1.IsNull(state1)) {
        IO::errorf("not enough memory: is the number of qubits too large?\n");
        return false;
      }

      rstate0 = &state0;
      rstate1 = &state1;
    }

    return true;
  }
};

}  // namespace qsim

#endif  // HYBRID_H_
