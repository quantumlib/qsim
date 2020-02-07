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

#ifndef FUSER_BASIC_H_
#define FUSER_BASIC_H_

#include <utility>
#include <vector>

#include "gate.h"
#include "fuser.h"

namespace qsim {

template <typename Gate>
struct BasicGateFuser final {
  using GateFused = qsim::GateFused<Gate>;

  static std::vector<GateFused> FuseGates(unsigned num_qubits,
      const std::vector<Gate>& gates, unsigned time) {
    std::vector<unsigned> times_to_split_at(1, time);
    return FuseGates(num_qubits, gates, times_to_split_at);
  }

  static std::vector<GateFused> FuseGates(
      unsigned num_qubits, const std::vector<Gate>& gates,
      const std::vector<unsigned>& times_to_split_at) {
    std::vector<GateFused> gates_fused;

    // Sequence of two-qubit gates and fixed single-qubit gates.
    std::vector<const Gate*> gates_seq;

    // Lattice of gates: qubits "hyperplane" and time direction.
    std::vector<std::vector<const Gate*>> gates_lat(num_qubits);

    // Current unfused gate index.
    std::size_t gate_index = 0;

    for (std::size_t l = 0; l < times_to_split_at.size(); ++l) {
      gates_seq.resize(0);
      gates_seq.reserve(gates.size());

      for (unsigned k = 0; k < num_qubits; ++k) {
        gates_lat[k].resize(0);
        gates_lat[k].reserve(128);
      }

      // Fill gates_seq and gates_lat in.
      for (; gate_index < gates.size(); ++gate_index) {
        const auto& gate = gates[gate_index];

        if (gates[gate_index].time > times_to_split_at[l]) break;

        if (gate.num_qubits == 1) {
          gates_lat[gate.qubits[0]].push_back(&gate);
          if (gate.unfusible) {
            gates_seq.push_back(&gate);
          }
        } else if (gate.num_qubits == 2) {
          gates_lat[gate.qubits[0]].push_back(&gate);
          gates_lat[gate.qubits[1]].push_back(&gate);
          gates_seq.push_back(&gate);
        }
      }

      std::vector<unsigned> last(num_qubits, 0);

      // Fuse gates.
      for (auto pgate : gates_seq) {
        if (pgate->num_qubits == 1) {
          unsigned q0 = pgate->qubits[0];

          GateFused gate_f = {pgate->kind, pgate->time, 1, {q0}, pgate};

          last[q0] = Advance(last[q0], gates_lat[q0], gate_f.gates);
          gate_f.gates.push_back(gates_lat[q0][last[q0]]);
          last[q0] = Advance(last[q0] + 1, gates_lat[q0], gate_f.gates);

          gates_fused.push_back(std::move(gate_f));
        } else if (pgate->num_qubits == 2) {
          unsigned q0 = pgate->qubits[0];
          unsigned q1 = pgate->qubits[1];

          if (Done(last[q0], pgate->time, gates_lat[q0])) continue;

          GateFused gate_f = {pgate->kind, pgate->time, 2, {q0, q1}, pgate};

          do {
            last[q0] = Advance(last[q0], gates_lat[q0], gate_f.gates);
            last[q1] = Advance(last[q1], gates_lat[q1], gate_f.gates);
            // Here gates_lat[q0][last[q0]] == gates_lat[q1][last[q1]].

            gate_f.gates.push_back(gates_lat[q0][last[q0]]);

            last[q0] = Advance(last[q0] + 1, gates_lat[q0], gate_f.gates);
            last[q1] = Advance(last[q1] + 1, gates_lat[q1], gate_f.gates);
          } while (NextGate(last[q0], gates_lat[q0], last[q1], gates_lat[q1]));

          gates_fused.push_back(std::move(gate_f));
        }
      }

      for (unsigned q = 0; q < num_qubits; ++q) {
        auto l = last[q];
        if (l == gates_lat[q].size()) continue;

        // Orphaned qubit.

        auto pgate = gates_lat[q][l];

        GateFused gate_f = {pgate->kind, pgate->time, 1, {q}, pgate};
        gate_f.gates.push_back(gates_lat[q][l]);

        l = Advance(l + 1, gates_lat[q], gate_f.gates);
        // Here l == gates_lat[q].size().

        gates_fused.push_back(std::move(gate_f));
      }
    }

    return gates_fused;
  }

 private:
  static unsigned Advance(unsigned k, const std::vector<const Gate*>& wl,
                     std::vector<const Gate*>& gates) {
    while (k < wl.size() && wl[k]->num_qubits == 1 && !wl[k]->unfusible) {
      gates.push_back(wl[k++]);
    }
    return k;
  }

  static bool Done(
      unsigned k, unsigned t, const std::vector<const Gate*>& wl) {
    return k >= wl.size() || wl[k]->time > t;
  }

  static bool NextGate(unsigned k1, const std::vector<const Gate*>& wl1,
                       unsigned k2, const std::vector<const Gate*>& wl2) {
    return k1 < wl1.size() && k2 < wl2.size() && wl1[k1] == wl2[k2];
  }
};

}  // namespace qsim

#endif  // FUSER_BASIC_H_
