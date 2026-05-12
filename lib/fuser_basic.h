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

#include "fuser.h"
#include "gate.h"
#include "operation.h"
#include "operation_base.h"

namespace qsim {

/**
 * Stateless object with methods for aggregating matrix `Gate`s into
 * `FusedGate`s. Measurement gates with equal times are fused together.
 * Non-matrix gates (such as user-defined controlled gates) and matrix-gates
 * acting on more than two qubits are not fused.
 * This class is deprecated. It is recommended to use MultiQubitGateFuser
 * from fuser_mqubit.h.
 */
template <typename IO>
class BasicGateFuser final : public Fuser<IO> {
 private:
  using Base = Fuser<IO>;

 public:
  /**
   * User-specified parameters for gate fusion.
   * BasicGateFuser does not use any parameters.
   */
  struct Parameter {
    unsigned verbosity = 0;
  };

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit matrix gates (represented by the Gate struct) are fused.
   * To respect specific time boundaries while fusing gates, use the other
   * version of this method below.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by
   *   operations.
   * @param ops Operations to be fused. Operation times for gates acting
   *   on the same qubits must be ordered. Operations that are out of time
   *   order must not cross the time boundaries set by `times_to_split_at`
   *   or by measurement gates.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused operations. Each element is a variant
   *   containing:
   *   - A `FusedGate` object for merged gates.
   *   - A `Measurement` object (representing a fused measurement).
   *   - A `const Operation*` pointing to the original operation if it was not
   *     eligible for fusion (e.g., multi-controlled gates).
   *   Note: The input operation container must outlive the returned vector,
   *   as the variant may hold pointers to the original operations.
   */
  template <typename Operation>
  static auto FuseGates(const Parameter& param, unsigned max_qubit1,
                        const std::vector<Operation>& ops,
                        bool fuse_matrix = true) {
    return FuseGates<Operation>(
        param, max_qubit1, ops.cbegin(), ops.cend(), {}, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit matrix gates (represented by the Gate struct) are fused.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by
   *   operations.
   * @param ops Operations to be fused. Operation times for gates acting
   *   on the same qubits must be ordered. Operations that are out of time
   *   order must not cross the time boundaries set by `times_to_split_at`
   *   or by measurement gates.
   * @param times_to_split_at Ordered list of time steps (boundaries) at which
   *   to separate fused gates. Each element of the output contains gates
   *   from a single 'window' defined by this list.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused operations. Each element is a variant
   *   containing:
   *   - A `FusedGate` object for merged gates.
   *   - A `Measurement` object (representing a fused measurement).
   *   - A `const Operation*` pointing to the original operation if it was not
   *     eligible for fusion (e.g., multi-controlled gates).
   *   Note: The input operation container must outlive the returned vector,
   *   as the variant may hold pointers to the original operations.
   */
  template <typename Operation>
  static auto FuseGates(const Parameter& param, unsigned max_qubit1,
                        const std::vector<Operation>& ops,
                        const std::vector<unsigned>& times_to_split_at,
                        bool fuse_matrix = true) {
    return FuseGates<Operation>(param, max_qubit1, ops.cbegin(), ops.cend(),
                                times_to_split_at, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit matrix gates (represented by the Gate struct) are fused.
   * To respect specific time boundaries while fusing gates, use the other
   * version of this method below.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by
   *   operations.
   * @param obeg, oend The iterator range [obeg, oend) of operations
   *   to be fused. Operation times for gates acting on the same qubits
   *   must be ordered. Operations that are out of time order must
   *   not cross the time boundaries set by `times_to_split_at` or
   *   by measurement gates.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused operations. Each element is a variant
   *   containing:
   *   - A `FusedGate` object for merged gates.
   *   - A `Measurement` object (representing a fused measurement).
   *   - A `const Operation*` pointing to the original operation if it was not
   *     eligible for fusion (e.g., multi-controlled gates).
   *   Note: The input operation container must outlive the returned vector,
   *   as the variant may hold pointers to the original operations.
   */
  template <typename Operation>
  static auto FuseGates(
      const Parameter& param, unsigned max_qubit1,
      typename std::vector<Operation>::const_iterator obeg,
      typename std::vector<Operation>::const_iterator oend,
      bool fuse_matrix = true) {
    return FuseGates<Operation>(
        param, max_qubit1, obeg, oend, {}, fuse_matrix);
  }

  /**
   * Stores sets of gates that can be applied together. Only one- and
   * two-qubit matrix gates (represented by the Gate struct) are fused.
   * @param param Options for gate fusion.
   * @param max_qubit1 The maximum qubit index (plus one) acted on by
   *   operations.
   * @param obeg, oend The iterator range [obeg, oend) of operations
   *   to be fused. Operation times for gates acting on the same qubits
   *   must be ordered. Operations that are out of time order must
   *   not cross the time boundaries set by `times_to_split_at` or
   *   by measurement gates.
   * @param times_to_split_at Ordered list of time steps (boundaries) at which
   *   to separate fused gates. Each element of the output contains gates
   *   from a single 'window' defined by this list.
   * @param fuse_matrix If true, multiply gate matrices together.
   * @return A vector of fused operations. Each element is a variant
   *   containing:
   *   - A `FusedGate` object for merged gates.
   *   - A `Measurement` object (representing a fused measurement).
   *   - A `const Operation*` pointing to the original operation if it was not
   *     eligible for fusion (e.g., multi-controlled gates).
   *   Note: The input operation container must outlive the returned vector,
   *   as the variant may hold pointers to the original operations.
   */
  template <typename OperationP>
  static auto FuseGates(
      const Parameter& param, unsigned max_qubit1,
      typename std::vector<OperationP>::const_iterator obeg,
      typename std::vector<OperationP>::const_iterator oend,
      const std::vector<unsigned>& times_to_split_at,
      bool fuse_matrix = true) {
    using Operation = std::remove_pointer_t<OperationP>;
    using fp_type = OpFpType<Operation>;
    using Gate = qsim::Gate<fp_type>;
    using ControlledGate = qsim::ControlledGate<fp_type>;
    using DecomposedGate = qsim::DecomposedGate<fp_type>;
    using FusedGate = qsim::FusedGate<fp_type>;
    using OperationF = std::variant<FusedGate, Measurement, const Operation*>;

    std::vector<OperationF> fused_ops;

    if (obeg >= oend) return fused_ops;

    std::size_t num_ops = oend - obeg;

    fused_ops.reserve(num_ops);

    // Merge with measurement gate times to separate fused gates at.
    auto times = Base::template MergeWithMeasurementTimes<OperationP>(
        obeg, oend, times_to_split_at);

    // Map to keep track of measurement gates with equal times.
    std::map<unsigned, std::vector<const Operation*>> measurement_gates;

    // Sequence of top level operations the other gates get fused to.
    std::vector<const Operation*> ops_seq;

    // Sequence of zero-qubit gates.
    std::vector<const Operation*> ops_seq0;

    // Lattice of operations: qubits "hyperplane" and time direction.
    std::vector<std::vector<const Operation*>> ops_lat(max_qubit1);

    // Current unfused operation.
    auto op_it = obeg;

    std::size_t last_fused_op_index = 0;

    // Iterate over time windows.
    for (std::size_t l = 0; l < times.size(); ++l) {
      ops_seq.resize(0);
      ops_seq.reserve(num_ops);

      ops_seq0.resize(0);
      ops_seq0.reserve(num_ops);

      for (unsigned k = 0; k < max_qubit1; ++k) {
        ops_lat[k].resize(0);
        ops_lat[k].reserve(128);
      }

      // Iterate over input operations and fill ops_seq and ops_lat in.
      for (; op_it < oend; ++op_it) {
        const auto& op = Base::OperationToConstRef(*op_it);
        const auto& bop = OpBaseOperation(op);

        if (bop.time > times[l]) break;

        if (!ValidateOp(op, bop, max_qubit1, ops_lat)) {
          fused_ops.resize(0);
          return fused_ops;
        }

        if (OpGetAlternative<Measurement>(op)) {
          auto& mea_gates_at_time = measurement_gates[bop.time];
          if (mea_gates_at_time.size() == 0) {
            ops_seq.push_back(&op);
            mea_gates_at_time.reserve(max_qubit1);
          }

          mea_gates_at_time.push_back(&op);
        } else {
          for (auto q : bop.qubits) {
            ops_lat[q].push_back(&op);
          }

          if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
            for (auto q : pg->controlled_by) {
              ops_lat[q].push_back(&op);
            }
          }

          bool is_gate = OpGetAlternative<Gate>(op) != nullptr;

          if (is_gate && bop.qubits.size() == 0) {
            ops_seq0.push_back(&op);
          } else if (!is_gate || bop.qubits.size() != 1) {
            ops_seq.push_back(&op);
          }
        }
      }

      std::vector<unsigned> last(max_qubit1, 0);

      const Operation* deferred_measurement = nullptr;

      // Fuse gates.
      for (auto pop : ops_seq) {
        const auto& qubits = OpQubits(*pop);

        if (OpGetAlternative<Measurement>(*pop)) {
          deferred_measurement = pop;
        } else if (const auto* pg = OpGetAlternative<DecomposedGate>(*pop)) {
          unsigned q0 = qubits[0];
          FusedGate fgate{pg->kind, pg->time, {q0}, pg, {}, {}};

          last[q0] = Advance(last[q0], ops_lat[q0], fgate.gates);

          // Here pop == ops_lat[q0][last[q0]].

          fgate.gates.push_back(pg);
          last[q0] = Advance(last[q0] + 1, ops_lat[q0], fgate.gates);

          fused_ops.push_back(std::move(fgate));
        } else if (!OpGetAlternative<Gate>(*pop) || qubits.size() > 2) {
          // Multi-qubit or controlled gate.

          for (auto q : qubits) {
            unsigned l = last[q];
            if (ops_lat[q][l] != pop) {
              last[q] = AddOrphanedQubit(q, l, ops_lat, fused_ops);
            }
            ++last[q];
          }

          if (const auto* pg = OpGetAlternative<ControlledGate>(*pop)) {
            for (auto q : pg->controlled_by) {
              unsigned l = last[q];
              if (ops_lat[q][l] != pop) {
                last[q] = AddOrphanedQubit(q, l, ops_lat, fused_ops);
              }
              ++last[q];
            }
          }

          fused_ops.push_back(pop);
        } else {
          // Two-qubit gate.

          unsigned q0 = qubits[0];
          unsigned q1 = qubits[1];

          const auto& gate = *OpGetAlternative<Gate>(*pop);

          if (Done(last[q0], gate.time, ops_lat[q0])) continue;

          FusedGate fgate = {gate.kind, gate.time, {q0, q1}, &gate, {}, {}};

          do {
            last[q0] = Advance(last[q0], ops_lat[q0], fgate.gates);
            last[q1] = Advance(last[q1], ops_lat[q1], fgate.gates);

            // Here ops_lat[q0][last[q0]] == ops_lat[q1][last[q1]].

            const auto& gate2 = *OpGetAlternative<Gate>(*ops_lat[q0][last[q0]]);
            fgate.gates.push_back(&gate2);

            last[q0] = Advance(last[q0] + 1, ops_lat[q0], fgate.gates);
            last[q1] = Advance(last[q1] + 1, ops_lat[q1], fgate.gates);
          } while (NextGate(last[q0], ops_lat[q0], last[q1], ops_lat[q1]));

          fused_ops.push_back(std::move(fgate));
        }
      }

      for (unsigned q = 0; q < max_qubit1; ++q) {
        auto l = last[q];
        if (l == ops_lat[q].size()) continue;

        // Orphaned qubit.
        AddOrphanedQubit(q, l, ops_lat, fused_ops);
      }

      if (deferred_measurement != nullptr) {
        const auto& mea_gates_at_time =
            measurement_gates[OpTime(*deferred_measurement)];

        if (mea_gates_at_time.size() == 1) {
          fused_ops.push_back(deferred_measurement);
        } else {
          Measurement mfused =
              *OpGetAlternative<Measurement>(*deferred_measurement);

          // Fuse measurement gates with equal times.
          for (const auto* pop : mea_gates_at_time) {
            if (pop == deferred_measurement) continue;

            const auto& qs = OpQubits(*pop);
            mfused.qubits.insert(mfused.qubits.end(), qs.begin(), qs.end());
          }

          fused_ops.push_back(std::move(mfused));
        }
      }

      if (ops_seq0.size() != 0) {
        Base::FuseZeroQubitGates(ops_seq0, [](const Operation* g) { return g; },
                                 last_fused_op_index, fused_ops);
      }

      if (op_it == oend) break;

      last_fused_op_index = fused_ops.size();
    }

    if (fuse_matrix) {
      for (auto& op : fused_ops) {
        if (auto* pg = OpGetAlternative<FusedGate>(op)) {
          if (!pg->ParentIsDecomposed()) {
            CalculateFusedMatrix(*pg);
          }
        }
      }
    }

    return fused_ops;
  }

 private:
  template <typename Operation, typename PGate>
  static unsigned Advance(unsigned k, const std::vector<const Operation*>& wl,
                          std::vector<PGate>& gates) {
    using Gate = qsim::Gate<OpFpType<Operation>>;

    while (k < wl.size()) {
      if (const auto* pg = OpGetAlternative<Gate>(*wl[k])) {
        if (pg->qubits.size() == 1) {
          gates.push_back(pg);
        } else {
          break;
        }
      } else {
        break;
      }
      ++k;
    }

    return k;
  }

  template <typename Operation>
  static bool Done(
      unsigned k, unsigned t, const std::vector<const Operation*>& wl) {
    return k >= wl.size() || OpTime(*wl[k]) > t;
  }

  template <typename Operation>
  static bool NextGate(unsigned k1, const std::vector<const Operation*>& wl1,
                       unsigned k2, const std::vector<const Operation*>& wl2) {
    return k1 < wl1.size() && k2 < wl2.size()
        && wl1[k1] == wl2[k2] && OpQubits(*wl1[k1]).size() < 3
        && OpGetAlternative<Gate<OpFpType<Operation>>>(*wl1[k1]);
  }

  template <typename OpsLat, typename OperationF>
  static unsigned AddOrphanedQubit(unsigned q, unsigned k,
                                   const OpsLat& ops_lat,
                                   std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using Gate = qsim::Gate<fp_type>;

    const auto& gate1 = *OpGetAlternative<Gate>(*ops_lat[q][k]);

    FusedGate fgate{gate1.kind, gate1.time, {q}, &gate1, {}, {}};
    fgate.gates.push_back(&gate1);

    k = Advance(k + 1, ops_lat[q], fgate.gates);

    fused_ops.push_back(std::move(fgate));

    return k;
  }

  template <typename OpsLat, typename Operation, typename BaseOperation>
  static bool ValidateOp(const Operation& op, const BaseOperation& bop,
                         unsigned max_qubit1, const OpsLat& ops_lat) {
    using ControlledGate = qsim::ControlledGate<OpFpType<Operation>>;

    for (unsigned q : bop.qubits) {
      if (q >= max_qubit1) {
        IO::errorf("fuser: gate qubit %u is out of range "
                   "(should be smaller than %u).\n", q, max_qubit1);
        return false;
      }
      if (!ops_lat[q].empty() && bop.time <= OpTime(*ops_lat[q].back())) {
        IO::errorf("fuser: gate at time %u is out of time order.\n", bop.time);
        return false;
      }
    }

    if (const auto& pg = OpGetAlternative<ControlledGate>(op)) {
      for (unsigned q : pg->controlled_by) {
        if (q >= max_qubit1) {
          IO::errorf("fuser: gate qubit %u is out of range "
                     "(should be smaller than %u).\n", q, max_qubit1);
          return false;
        }
        if (!ops_lat[q].empty() && bop.time <= OpTime(*ops_lat[q].back())) {
          IO::errorf(
              "fuser: gate at time %u is out of time order.\n", bop.time);
          return false;
        }
      }
    }

    return true;
  }
};

}  // namespace qsim

#endif  // FUSER_BASIC_H_
