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

#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "gate.h"
#include "fuser.h"

namespace qsim {

/**
 * Stateless object with methods for aggregating `Gate`s into `GateFused`.
 * Measurement gates with equal times are fused together.
 * User-defined controlled gates (controlled_by.size() > 0) and gates acting on
 * more than two qubits are not fused.
 * The template parameter Gate can be Gate type or a pointer to Gate type.
 * This class is deprecated. It is recommended to use MultiQubitGateFuser
 * from fuser_mqubit.h.
 */
template <typename IO, typename Gate>
class BasicGateFuser final : public Fuser<IO, Gate> {
 private:
  using Base = Fuser<IO, Gate>;
  using RGate = typename Base::RGate;

 public:
  using GateFused = qsim::GateFused<RGate>;

  /**
   * User-specified parameters for gate fusion.
   * BasicGateFuser does not use any parameters.
   */
  struct Parameter {
    unsigned verbosity = 0;
  };

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit gates will get fused. To respect specific time boundaries while
   * fusing gates, use the other version of this method below.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by 'gates'.
   * @param gates The gates (or pointers to the gates) to be fused.
   *   Gate times of the gates that act on the same qubits should be ordered.
   *   Gates that are out of time order should not cross the time boundaries
   *   set by measurement gates.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(const Parameter& param,
                                          unsigned max_qubit1,
                                          const std::vector<Gate>& gates,
                                          bool fuse_matrix = true) {
    return FuseGates(
        param, max_qubit1, gates.cbegin(), gates.cend(), {}, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit gates will get fused.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by 'gates'.
   * @param gates The gates (or pointers to the gates) to be fused.
   *   Gate times of the gates that act on the same qubits should be ordered.
   *   Gates that are out of time order should not cross the time boundaries
   *   set by `times_to_split_at` or by measurment gates.
   * @param times_to_split_at Ordered list of time steps (boundaries) at which
   *   to separate fused gates. Each element of the output will contain gates
   *   from a single 'window' in this list.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param,
      unsigned max_qubit1, const std::vector<Gate>& gates,
      const std::vector<unsigned>& times_to_split_at,
      bool fuse_matrix = true) {
    return FuseGates(param, max_qubit1, gates.cbegin(), gates.cend(),
                     times_to_split_at, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit gates will get fused. To respect specific time boundaries while
   * fusing gates, use the other version of this method below.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by 'gates'.
   * @param gfirst, glast The iterator range [gfirst, glast) to fuse gates
   *   (or pointers to gates) in. Gate times of the gates that act on the same
   *   qubits should be ordered. Gates that are out of time order should not
   *   cross the time boundaries set by measurement gates.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param, unsigned max_qubit1,
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      bool fuse_matrix = true) {
    return FuseGates(param, max_qubit1, gfirst, glast, {}, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit gates will get fused.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by 'gates'.
   * @param gfirst, glast The iterator range [gfirst, glast) to fuse gates
   *   (or pointers to gates) in. Gate times of the gates that act on the same
   *   qubits should be ordered. Gates that are out of time order should not
   *   cross the time boundaries set by `times_to_split_at` or by measurment
   *   gates.
   * @param times_to_split_at Ordered list of time steps (boundaries) at which
   *   to separate fused gates. Each element of the output will contain gates
   *   from a single 'window' in this list.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param, unsigned max_qubit1,
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      const std::vector<unsigned>& times_to_split_at,
      bool fuse_matrix = true) {
    std::vector<GateFused> gates_fused;

    if (gfirst >= glast) return gates_fused;

    std::size_t num_gates = glast - gfirst;

    gates_fused.reserve(num_gates);

    // Merge with measurement gate times to separate fused gates at.
    auto times =
        Base::MergeWithMeasurementTimes(gfirst, glast, times_to_split_at);

    // Map to keep track of measurement gates with equal times.
    std::map<unsigned, std::vector<const RGate*>> measurement_gates;

    // Sequence of top level gates the other gates get fused to.
    std::vector<const RGate*> gates_seq;

    // Lattice of gates: qubits "hyperplane" and time direction.
    std::vector<std::vector<const RGate*>> gates_lat(max_qubit1);

    // Current unfused gate.
    auto gate_it = gfirst;

    for (std::size_t l = 0; l < times.size(); ++l) {
      gates_seq.resize(0);
      gates_seq.reserve(num_gates);

      for (unsigned k = 0; k < max_qubit1; ++k) {
        gates_lat[k].resize(0);
        gates_lat[k].reserve(128);
      }

      // Fill gates_seq and gates_lat in.
      for (; gate_it < glast; ++gate_it) {
        const auto& gate = Base::GateToConstRef(*gate_it);

        if (gate.time > times[l]) break;

        if (!ValidateGate(gate, max_qubit1, gates_lat)) {
          gates_fused.resize(0);
          return gates_fused;
        }

        if (gate.kind == gate::kMeasurement) {
          auto& mea_gates_at_time = measurement_gates[gate.time];
          if (mea_gates_at_time.size() == 0) {
            gates_seq.push_back(&gate);
            mea_gates_at_time.reserve(max_qubit1);
          }

          mea_gates_at_time.push_back(&gate);
        } else if (gate.controlled_by.size() > 0 || gate.qubits.size() > 2) {
          for (auto q : gate.qubits) {
            gates_lat[q].push_back(&gate);
          }
          for (auto q : gate.controlled_by) {
            gates_lat[q].push_back(&gate);
          }
          gates_seq.push_back(&gate);
        } else if (gate.qubits.size() == 1) {
          gates_lat[gate.qubits[0]].push_back(&gate);
          if (gate.unfusible) {
            gates_seq.push_back(&gate);
          }
        } else if (gate.qubits.size() == 2) {
          gates_lat[gate.qubits[0]].push_back(&gate);
          gates_lat[gate.qubits[1]].push_back(&gate);
          gates_seq.push_back(&gate);
        }
      }

      std::vector<unsigned> last(max_qubit1, 0);

      const RGate* delayed_measurement_gate = nullptr;

      // Fuse gates.
      for (auto pgate : gates_seq) {
        if (pgate->kind == gate::kMeasurement) {
          delayed_measurement_gate = pgate;
        } else if (pgate->qubits.size() > 2
                   || pgate->controlled_by.size() > 0) {
          // Multi-qubit or controlled gate.

          for (auto q : pgate->qubits) {
            unsigned l = last[q];
            if (gates_lat[q][l] != pgate) {
              last[q] = AddOrphanedQubit(q, l, gates_lat, gates_fused);
            }
            ++last[q];
          }

          for (auto q : pgate->controlled_by) {
            unsigned l = last[q];
            if (gates_lat[q][l] != pgate) {
              last[q] = AddOrphanedQubit(q, l, gates_lat, gates_fused);
            }
            ++last[q];
          }

          gates_fused.push_back({pgate->kind, pgate->time, pgate->qubits,
                                 pgate, {pgate}, {}});
        } else if (pgate->qubits.size() == 1) {
          unsigned q0 = pgate->qubits[0];

          GateFused gate_f = {pgate->kind, pgate->time, {q0}, pgate, {}, {}};

          last[q0] = Advance(last[q0], gates_lat[q0], gate_f.gates);
          gate_f.gates.push_back(gates_lat[q0][last[q0]]);
          last[q0] = Advance(last[q0] + 1, gates_lat[q0], gate_f.gates);

          gates_fused.push_back(std::move(gate_f));
        } else if (pgate->qubits.size() == 2) {
          unsigned q0 = pgate->qubits[0];
          unsigned q1 = pgate->qubits[1];

          if (Done(last[q0], pgate->time, gates_lat[q0])) continue;

          GateFused gate_f =
              {pgate->kind, pgate->time, {q0, q1}, pgate, {}, {}};

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

      for (unsigned q = 0; q < max_qubit1; ++q) {
        auto l = last[q];
        if (l == gates_lat[q].size()) continue;

        // Orphaned qubit.
        AddOrphanedQubit(q, l, gates_lat, gates_fused);
      }

      if (delayed_measurement_gate != nullptr) {
        auto pgate = delayed_measurement_gate;

        const auto& mea_gates_at_time = measurement_gates[pgate->time];

        GateFused gate_f = {pgate->kind, pgate->time, {}, pgate, {}, {}};
        gate_f.gates.reserve(mea_gates_at_time.size());

        // Fuse measurement gates with equal times.

        for (const auto* pgate : mea_gates_at_time) {
          gate_f.qubits.insert(gate_f.qubits.end(),
                               pgate->qubits.begin(), pgate->qubits.end());
          gate_f.gates.push_back(pgate);
        }

        gates_fused.push_back(std::move(gate_f));
      }

      if (gate_it == glast) break;
    }

    if (fuse_matrix) {
      for (auto& gate_f : gates_fused) {
        if (gate_f.kind != gate::kMeasurement && gate_f.kind != gate::kDecomp) {
          CalculateFusedMatrix(gate_f);
        }
      }
    }

    return gates_fused;
  }

 private:
  static unsigned Advance(unsigned k, const std::vector<const RGate*>& wl,
                          std::vector<const RGate*>& gates) {
    while (k < wl.size() && wl[k]->qubits.size() == 1
           && wl[k]->controlled_by.size() == 0 && !wl[k]->unfusible) {
      gates.push_back(wl[k++]);
    }

    return k;
  }

  static bool Done(
      unsigned k, unsigned t, const std::vector<const RGate*>& wl) {
    return k >= wl.size() || wl[k]->time > t;
  }

  static bool NextGate(unsigned k1, const std::vector<const RGate*>& wl1,
                       unsigned k2, const std::vector<const RGate*>& wl2) {
    return k1 < wl1.size() && k2 < wl2.size() && wl1[k1] == wl2[k2]
        && wl1[k1]->qubits.size() < 3 && wl1[k1]->controlled_by.size() == 0;
  }

  template <typename GatesLat>
  static unsigned AddOrphanedQubit(unsigned q, unsigned k,
                                   const GatesLat& gates_lat,
                                   std::vector<GateFused>& gates_fused) {
    auto pgate = gates_lat[q][k];

    GateFused gate_f = {pgate->kind, pgate->time, {q}, pgate, {}, {}};
    gate_f.gates.push_back(pgate);

    k = Advance(k + 1, gates_lat[q], gate_f.gates);

    gates_fused.push_back(std::move(gate_f));

    return k;
  }

  template <typename Gate2, typename GatesLat>
  static bool ValidateGate(const Gate2& gate, unsigned max_qubit1,
                           const GatesLat& gates_lat) {
    for (unsigned q : gate.qubits) {
      if (q >= max_qubit1) {
        IO::errorf("fuser: gate qubit %u is out of range "
                   "(should be smaller than %u).\n", q, max_qubit1);
        return false;
      }
      if (!gates_lat[q].empty() && gate.time <= gates_lat[q].back()->time) {
        IO::errorf("fuser: gate at time %u is out of time order.\n", gate.time);
        return false;
      }
    }

    for (unsigned q : gate.controlled_by) {
      if (q >= max_qubit1) {
        IO::errorf("fuser: gate qubit %u is out of range "
                   "(should be smaller than %u).\n", q, max_qubit1);
        return false;
      }
      if (!gates_lat[q].empty() && gate.time <= gates_lat[q].back()->time) {
        IO::errorf("fuser: gate at time %u is out of time order.\n", gate.time);
        return false;
      }
    }

    return true;
  }
};

}  // namespace qsim

#endif  // FUSER_BASIC_H_
