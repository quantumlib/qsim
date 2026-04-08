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

#ifndef FUSER_MQUBIT_H_
#define FUSER_MQUBIT_H_

#include <algorithm>
#include <cstdint>
#include <limits>
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
 * Non-matrix gates (such as user-defined controlled gates) are not fused.
 * This fuser can fuse into up to six-qbuit gates.
 */
template <typename IO>
class MultiQubitGateFuser final : public Fuser<IO> {
 private:
  using Base = Fuser<IO>;

  // Auxillary classes and structs.

  // Manages doubly-linked lists.
  template <typename T>
  class LinkManagerT {
   public:
    struct Link {
      T val;
      Link* next;
      Link* prev;
    };

    explicit LinkManagerT(uint64_t size) {
      links_.reserve(size);
    }

    Link* AddBack(const T& t, Link* link) {
      if (link == nullptr) {
        links_.push_back({t, nullptr, nullptr});
      } else {
        links_.push_back({t, link->next, link});
        link->next = &links_.back();
      }

      return &links_.back();
    }

    static void Delete(const Link* link) {
      if (link->prev != nullptr) {
        link->prev->next = link->next;
      }
      if (link->next != nullptr) {
        link->next->prev = link->prev;
      }
    }

   private:
    std::vector<Link> links_;
  };

  template <typename Parent, typename PGate>
  struct GateF;

  template <typename Parent, typename PGate>
  using LinkManager = LinkManagerT<GateF<Parent, PGate>*>;
  template <typename Parent, typename PGate>
  using Link = typename LinkManager<Parent, PGate>::Link;

  // Intermediate representation of a fused gate.
  template <typename Parent, typename PGate>
  struct GateF {
    using fp_type = OpFpType<Parent>;

    const Parent* parent;
    Qubits qubits;
    std::vector<PGate> gates;  // Gates that get fused to this gate.
    std::vector<Link<Parent, PGate>*> links;  // Gate "lattice" links.
    uint64_t mask;                            // Qubit mask.
    unsigned visited;
  };

  // Possible values for visited in GateF.
  // Note that MakeGateSequence assignes values from kSecond to the number of
  // gates in the sequence plus one, see below.
  enum Visited {
    kZero = 0,              // Start value for matrix gates.
    kFirst = 1,             // Value after the first pass for partially fused
                            // matrix gates.
    kSecond = 2,            // Start value to assign values in MakeGateSequence.
    kCompress = 99999997,   // Used to compress links.
    kUnfusible = 99999998,  // Start value for controlled or measurement gates.
    kFinal = 99999999,      // Value after the first pass for child gates or
                            // after the second pass for fused matrix gates,
                            // controlled gates, and measurement gates.
  };

  struct Stat {
    unsigned num_measurements = 0;
    unsigned num_fused_measurements = 0;
    unsigned num_fused_gates = 0;
    unsigned num_unfusible_gates = 0;
    std::vector<unsigned> num_gates;
  };

  // Gate that is added to a sequence of gates to fuse together.
  template <typename Parent, typename PGate>
  struct GateA {
    GateF<Parent, PGate>* gate;
    Qubits qubits;                            // Added qubits.
    std::vector<Link<Parent, PGate>*> links;  // Added lattice links.
  };

  template <typename Parent, typename PGate>
  struct Scratch {
    std::vector<GateA<Parent, PGate>> data;
    std::vector<GateA<Parent, PGate>*> prev1;
    std::vector<GateA<Parent, PGate>*> prev2;
    std::vector<GateA<Parent, PGate>*> next1;
    std::vector<GateA<Parent, PGate>*> next2;
    std::vector<GateA<Parent, PGate>*> longest_seq;
    std::vector<GateA<Parent, PGate>*> stack;
    std::vector<GateF<Parent, PGate>*> gates;
    unsigned count = 0;
  };

 public:
  /**
   * User-specified parameters for gate fusion.
   */
  struct Parameter {
    /**
     * Maximum number of qubits in a fused gate. It can take values from 2 to
     * 6 (0 and 1 are equivalent to 2). For optimal performance, a value of
     * 4 or 5 is recommended to balance memory bandwidth and computational
     * overhead.
     */
    unsigned max_fused_size = 2;
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
    using FusedGate = qsim::FusedGate<fp_type>;
    using PGate = typename FusedGate::PGate;
    using OperationF = std::variant<FusedGate, Measurement, const Operation*>;

    std::vector<OperationF> fused_ops;

    if (obeg >= oend) return fused_ops;

    std::size_t num_ops = oend - obeg;

    fused_ops.reserve(num_ops);

    // Merge with measurement gate times to separate fused gates at.
    auto times = Base::template MergeWithMeasurementTimes<OperationP>(
        obeg, oend, times_to_split_at);

    using GateF = MultiQubitGateFuser::GateF<Operation, PGate>;

    LinkManager<Operation, PGate> link_manager(max_qubit1 * num_ops);

    // Auxillary data structures.
    // Sequence of intermediate fused gates.
    std::vector<GateF> gates_seq;
    // Gate "lattice".
    std::vector<Link<Operation, PGate>*> gates_lat;
    // Sequences of intermediate fused gates ordered by gate size.
    std::vector<std::vector<GateF*>> fgates(max_qubit1 + 1);

    gates_seq.reserve(num_ops);
    gates_lat.reserve(max_qubit1);

    Scratch<Operation, PGate> scratch;

    scratch.data.reserve(1024);
    scratch.prev1.reserve(32);
    scratch.prev2.reserve(32);
    scratch.next1.reserve(32);
    scratch.next2.reserve(32);
    scratch.longest_seq.reserve(8);
    scratch.stack.reserve(8);

    Stat stat;
    stat.num_gates.resize(max_qubit1 + 1, 0);

    unsigned max_fused_size = std::min(unsigned{6}, param.max_fused_size);
    max_fused_size = std::min(max_fused_size, max_qubit1);

    std::size_t last_fused_gate_index = 0;
    auto op_it = obeg;

    // Iterate over time windows.
    for (std::size_t l = 0; l < times.size(); ++l) {
      gates_seq.resize(0);
      gates_lat.resize(0);
      gates_lat.resize(max_qubit1, nullptr);

      for (unsigned i = 0; i <= max_qubit1; ++i) {
        fgates[i].resize(0);
      }

      uint64_t max_gate_size = 0;
      GateF* last_measurement = nullptr;

      // Iterate over input operations.
      for (; op_it < oend; ++op_it) {
        const auto& op = Base::OperationToConstRef(*op_it);
        const auto& bop = OpBaseOperation(op);

        if (bop.time > times[l]) break;

        if (!ValidateOp(op, max_qubit1, gates_lat)) {
          fused_ops.resize(0);
          return fused_ops;
        }

        // Fill in auxillary data structures.

        if (OpGetAlternative<Measurement>(op)) {
          // Measurement gate.

          if (last_measurement == nullptr
              || OpTime(*last_measurement->parent) != bop.time) {
            gates_seq.push_back({&op, {}, {}, {}, 0, kUnfusible});
            last_measurement = &gates_seq.back();

            last_measurement->qubits.reserve(max_qubit1);
            last_measurement->links.reserve(max_qubit1);

            ++stat.num_fused_measurements;
          }

          for (auto q : bop.qubits) {
            last_measurement->qubits.push_back(q);
            last_measurement->mask |= uint64_t{1} << q;
            gates_lat[q] = link_manager.AddBack(last_measurement, gates_lat[q]);
            last_measurement->links.push_back(gates_lat[q]);
          }

          ++stat.num_measurements;
        } else {
          gates_seq.push_back({&op, {}, {}, {}, 0, kZero});
          auto& fgate = gates_seq.back();

          unsigned num_gate_qubits = bop.qubits.size();

          if (OpGetAlternative<Gate>(op)) {
            // Matrix gates that are fused.

            if (max_gate_size < num_gate_qubits) {
              max_gate_size = num_gate_qubits;
            }

            unsigned size = std::max(max_fused_size, num_gate_qubits);

            fgate.qubits.reserve(size);
            fgate.links.reserve(size);
            fgate.gates.reserve(4 * size);
            fgate.links.reserve(size);

            if (fgates[num_gate_qubits].empty()) {
              fgates[num_gate_qubits].reserve(num_ops);
            }
            fgates[num_gate_qubits].push_back(&fgate);

            for (auto q : bop.qubits) {
              fgate.qubits.push_back(q);
            }

            ++stat.num_gates[num_gate_qubits];
          } else {
            // Other gates are not fused.

            uint64_t size = num_gate_qubits;

            if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
              size += pg->controlled_by.size();
            }

            fgate.links.reserve(size);
            fgate.visited = kUnfusible;

            ++stat.num_unfusible_gates;
          }

          for (auto q : bop.qubits) {
            fgate.mask |= uint64_t{1} << q;
            gates_lat[q] = link_manager.AddBack(&fgate, gates_lat[q]);
            fgate.links.push_back(gates_lat[q]);
          }

          if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
            for (auto q : pg->controlled_by) {
              fgate.mask |= uint64_t{1} << q;
              gates_lat[q] = link_manager.AddBack(&fgate, gates_lat[q]);
              fgate.links.push_back(gates_lat[q]);
            }
          }
        }
      }

      // Fuse large gates with smaller gates.
      FuseGates<Gate>(max_gate_size, fgates);

      if (max_fused_size > 2) {
        FuseGateSequences(
            max_fused_size, max_qubit1, scratch, gates_seq, stat, fused_ops);
      } else {
        unsigned prev_time = 0;

        std::vector<GateF*> orphaned_gates;
        orphaned_gates.reserve(max_qubit1);

        for (auto& fgate : gates_seq) {
          if (fgate.visited != kUnfusible && fgate.gates.size() == 0) continue;

          unsigned time = OpTime(*fgate.parent);

          if (prev_time != time) {
            if (orphaned_gates.size() > 0) {
              FuseOrphanedGates(
                  max_fused_size, stat, orphaned_gates, fused_ops);
              orphaned_gates.resize(0);
            }

            prev_time = time;
          }

          if (fgate.qubits.size() == 1 && max_fused_size > 1
              && fgate.visited != kUnfusible) {
            orphaned_gates.push_back(&fgate);
            continue;
          }

          if (fgate.visited == kUnfusible) {
            AddUnfusible(fgate, fused_ops);
          } else {
            // Assume fgate.qubits (gate.qubits) are sorted.
            const Gate& parent = *OpGetAlternative<Gate>(*fgate.parent);
            fused_ops.push_back(FusedGate{parent.kind, parent.time,
                                          std::move(fgate.qubits), &parent,
                                          std::move(fgate.gates), {}});

            ++stat.num_fused_gates;
          }
        }

        if (orphaned_gates.size() > 0) {
          FuseOrphanedGates(max_fused_size, stat, orphaned_gates, fused_ops);
        }
      }

      if (fgates[0].size() != 0) {
        Base::FuseZeroQubitGates(fgates[0],
                                 [](const GateF* g) { return g->parent; },
                                 last_fused_gate_index, fused_ops);
      }

      last_fused_gate_index = fused_ops.size();
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

    PrintStat(param.verbosity, stat, fused_ops);

    return fused_ops;
  }

 private:
  // Fuse large gates with smaller gates.
  template <typename Gate, typename GateF>
  static void FuseGates(uint64_t max_gate_size,
                        std::vector<std::vector<GateF*>>& fgates) {
    // Traverse gates in order of decreasing size.
    for (uint64_t i = 0; i < max_gate_size; ++i) {
      std::size_t pos = 0;

      for (auto fgate : fgates[max_gate_size - i]) {
        if (fgate->visited > kZero) {
          if (fgate->gates.size() == 0) {
            fgate->visited = kFinal;
          }
          continue;
        }

        fgates[max_gate_size - i][pos++] = fgate;

        fgate->visited = kFirst;

        FusePrev(0, *fgate);
        fgate->gates.push_back(OpGetAlternative<Gate>(*fgate->parent));
        FuseNext(0, *fgate);
      }

      fgates[max_gate_size - i].resize(pos);
    }
  }

  // Try to fuse gate sequences as follows. Gate time goes from bottom to top.
  // Gates are fused either from left to right or from right to left.
  //
  // max_fused_size = 3: _-  or  -_
  //
  // max_fused_size = 4: _-_
  //
  // max_fused_size = 5: _-_-  or  -_-_
  //
  // max_fused_size = 6: _-_-_
  template <typename Scratch, typename GateF, typename OperationF>
  static void FuseGateSequences(unsigned max_fused_size,
                                unsigned max_qubit1, Scratch& scratch,
                                std::vector<GateF>& gates_seq, Stat& stat,
                                std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using Gate = qsim::Gate<fp_type>;

    unsigned prev_time = 0;

    std::vector<GateF*> orphaned_gates;
    orphaned_gates.reserve(max_qubit1);

    for (auto& fgate : gates_seq) {
      unsigned time = OpTime(*fgate.parent);

      if (prev_time != time) {
        if (orphaned_gates.size() > 0) {
          FuseOrphanedGates(max_fused_size, stat, orphaned_gates, fused_ops);
          orphaned_gates.resize(0);
        }

        prev_time = time;
      }

      if (fgate.visited == kFinal) continue;

      if (fgate.visited == kUnfusible) {
        AddUnfusible(fgate, fused_ops);

        continue;
      }

      if (fgate.qubits.size() >= max_fused_size) {
        const Gate& parent = *OpGetAlternative<Gate>(*fgate.parent);
        fused_ops.push_back(FusedGate{parent.kind, parent.time,
                                      std::move(fgate.qubits), &parent,
                                      std::move(fgate.gates), {}});

        fgate.visited = kFinal;
        ++stat.num_fused_gates;

        continue;
      }


      if (fgate.qubits.size() == 1 && max_fused_size > 1) {
        orphaned_gates.push_back(&fgate);
        continue;
      }

      scratch.data.resize(0);
      scratch.gates.resize(0);
      scratch.count = 0;

      MakeGateSequence(max_fused_size, scratch, fgate);

      if (scratch.gates.size() == 0) {
        orphaned_gates.push_back(&fgate);
      } else {
        for (auto fgate : scratch.gates) {
          std::sort(fgate->qubits.begin(), fgate->qubits.end());

          const Gate& parent = *OpGetAlternative<Gate>(*fgate->parent);
          fused_ops.push_back(FusedGate{parent.kind, parent.time,
                                        std::move(fgate->qubits), &parent,
                                        std::move(fgate->gates), {}});

          ++stat.num_fused_gates;
        }
      }
    }

    if (orphaned_gates.size() > 0) {
      FuseOrphanedGates(max_fused_size, stat, orphaned_gates, fused_ops);
    }
  }

  template <typename GateF, typename OperationF>
  static void FuseOrphanedGates(unsigned max_fused_size, Stat& stat,
                                std::vector<GateF*>& orphaned_gates,
                                std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using Gate = qsim::Gate<fp_type>;

    for (std::size_t i = 0; i < orphaned_gates.size(); ++i) {
      auto ogate1 = orphaned_gates[i];

      if (ogate1->visited == kFinal) continue;

      ogate1->visited = kFinal;

      for (std::size_t j = i + 1; j < orphaned_gates.size(); ++j) {
        auto ogate2 = orphaned_gates[j];

        if (ogate2->visited == kFinal) continue;

        unsigned cur_size = ogate1->qubits.size() + ogate2->qubits.size();

        if (cur_size <= max_fused_size) {
          ogate2->visited = kFinal;

          for (auto q : ogate2->qubits) {
            ogate1->qubits.push_back(q);
            ogate1->mask |= uint64_t{1} << q;
          }

          for (auto l : ogate2->links) {
            ogate1->links.push_back(l);
          }

          for (auto gate : ogate2->gates) {
            ogate1->gates.push_back(gate);
          }
        }

        if (cur_size == max_fused_size) {
          break;
        }
      }

      FuseNext(1, *ogate1);

      std::sort(ogate1->qubits.begin(), ogate1->qubits.end());

      const Gate& parent = *OpGetAlternative<Gate>(*ogate1->parent);
      fused_ops.push_back(FusedGate{parent.kind, parent.time,
                                    std::move(ogate1->qubits), &parent,
                                    std::move(ogate1->gates), {}});

      ++stat.num_fused_gates;
    }
  }

  template <typename GateF, typename OperationF>
  static void AddUnfusible(const GateF& fgate,
                           std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using DecomposedGate = qsim::DecomposedGate<fp_type>;

    if (const auto& pg = OpGetAlternative<Measurement>(*fgate.parent)) {
      if (pg->qubits.size() == fgate.qubits.size()) {
        fused_ops.push_back(fgate.parent);
      } else {
        Measurement mfused = *pg;
        mfused.qubits = fgate.qubits;
        fused_ops.push_back(std::move(mfused));
      }
    } else {
      if (const auto* pg = OpGetAlternative<DecomposedGate>(*fgate.parent)) {
        fused_ops.push_back(
            FusedGate{pg->kind, pg->time, {pg->qubits[0]}, pg, {pg}, {}});
      } else {
        fused_ops.push_back(fgate.parent);
      }
    }
  }

  template <typename Scratch, typename Parent, typename PGate>
  static void MakeGateSequence(unsigned max_fused_size,
                               Scratch& scratch, GateF<Parent, PGate>& fgate) {
    unsigned level = kSecond + scratch.count;

    FindLongestGateSequence(max_fused_size, level, scratch, fgate);

    auto longest_seq = scratch.longest_seq;

    if (longest_seq.size() == 1 && scratch.count == 0) {
      fgate.visited = kFirst;
      return;
    }

    ++scratch.count;

    for (auto p : longest_seq) {
      p->gate->visited = kCompress;

      for (auto q : p->qubits) {
        fgate.qubits.push_back(q);
        fgate.mask |= uint64_t{1} << q;
      }

      for (auto l : p->links) {
        fgate.links.push_back(l);
      }
    }

    // Compress links.
    for (auto& link : fgate.links) {
      while (link->prev != nullptr && link->prev->val->visited == kCompress) {
        link = link->prev;
      }

      while (link->next != nullptr && link->next->val->visited == kCompress) {
        LinkManager<Parent, PGate>::Delete(link->next);
      }
    }

    for (auto p : longest_seq) {
      p->gate->visited = level;
    }

    if (longest_seq.size() >= 3) {
      AddGatesFromNext(longest_seq[2]->gate->gates, fgate);
    }

    if (longest_seq.size() >= 5) {
      AddGatesFromNext(longest_seq[4]->gate->gates, fgate);
    }

    if (longest_seq.size() >= 2) {
      // May call MakeGateSequence recursively.
      AddGatesFromPrev(max_fused_size, *longest_seq[1]->gate, scratch, fgate);
    }

    if (longest_seq.size() >= 4) {
      // May call MakeGateSequence recursively.
      AddGatesFromPrev(max_fused_size, *longest_seq[3]->gate, scratch, fgate);
    }

    for (auto p : longest_seq) {
      p->gate->visited = kFinal;
    }

    FuseNext(1, fgate);

    scratch.gates.push_back(&fgate);
  }

  template <typename PGate, typename GateF>
  static void AddGatesFromNext(std::vector<PGate>& gates, GateF& fgate) {
    for (auto gate : gates) {
      fgate.gates.push_back(gate);
    }
  }

  template <typename GateF, typename Scratch>
  static void AddGatesFromPrev(unsigned max_fused_size, const GateF& pfgate,
                               Scratch& scratch, GateF& fgate) {
    for (auto gate : pfgate.gates) {
        fgate.gates.push_back(gate);
    }

    for (auto link : pfgate.links) {
      if (link->prev == nullptr) continue;

      auto pgate = link->prev->val;

      if (pgate->visited == kFirst) {
        MakeGateSequence(max_fused_size, scratch, *pgate);
      }
    }
  }

  template <typename Scratch, typename GateF>
  static void FindLongestGateSequence(unsigned max_fused_size, unsigned level,
                                      Scratch& scratch, GateF& fgate) {
    scratch.data.push_back({&fgate, {}, {}});

    scratch.longest_seq.resize(0);
    scratch.longest_seq.push_back(&scratch.data.back());

    scratch.stack.resize(0);
    scratch.stack.push_back(&scratch.data.back());

    unsigned cur_size = fgate.qubits.size();
    fgate.visited = level;

    unsigned max_size = cur_size;

    GetNextAvailableGates(max_fused_size, cur_size, fgate,
                          (const GateF*) nullptr,
                          scratch.data, scratch.next1);

    for (auto n1 : scratch.next1) {
      unsigned cur_size2 = cur_size + n1->qubits.size();
      if (cur_size2 > max_fused_size) continue;

      bool feasible = GetPrevAvailableGates(max_fused_size, cur_size,
                                            level, *n1->gate,
                                            (const GateF*) nullptr,
                                            scratch.data, scratch.prev1);

      if (!feasible) continue;

      if (scratch.prev1.size() == 0 && max_fused_size > 3) continue;

      if (cur_size2 == max_fused_size) {
        std::swap(scratch.longest_seq, scratch.stack);
        scratch.longest_seq.push_back(n1);
        return;
      }

      Push(level, cur_size2, cur_size, max_size, scratch, n1);

      for (auto p1 : scratch.prev1) {
        unsigned cur_size2 = cur_size + p1->qubits.size();

        if (cur_size2 > max_fused_size) {
          continue;
        } else if (cur_size2 == max_fused_size) {
          std::swap(scratch.longest_seq, scratch.stack);
          scratch.longest_seq.push_back(p1);
          return;
        }

        Push(level, cur_size2, cur_size, max_size, scratch, p1);

        GetNextAvailableGates(max_fused_size, cur_size, *p1->gate, &fgate,
                              scratch.data, scratch.next2);

        for (auto n2 : scratch.next2) {
          unsigned cur_size2 = cur_size + n2->qubits.size();
          if (cur_size2 > max_fused_size) continue;

          bool feasible = GetPrevAvailableGates(max_fused_size, cur_size,
                                                level, *n2->gate, n1->gate,
                                                scratch.data, scratch.prev2);

          if (!feasible) continue;

          if (cur_size2 == max_fused_size) {
            std::swap(scratch.longest_seq, scratch.stack);
            scratch.longest_seq.push_back(n2);
            return;
          }

          Push(level, cur_size2, cur_size, max_size, scratch, n2);

          for (auto p2 : scratch.prev2) {
            unsigned cur_size2 = cur_size + p2->qubits.size();

            if (cur_size2 > max_fused_size) {
              continue;
            } else if (cur_size2 == max_fused_size) {
              std::swap(scratch.longest_seq, scratch.stack);
              scratch.longest_seq.push_back(p2);
              return;
            }

            if (cur_size2 > max_size) {
              scratch.stack.push_back(p2);
              scratch.longest_seq = scratch.stack;
              scratch.stack.pop_back();
              max_size = cur_size2;
            }
          }

          Pop(cur_size, scratch, n2);
        }

        Pop(cur_size, scratch, p1);
      }

      Pop(cur_size, scratch, n1);
    }
  }

  template <typename Scratch, typename GateA>
  static void Push(unsigned level, unsigned cur_size2, unsigned& cur_size,
                   unsigned& max_size, Scratch& scratch, GateA* agate) {
    agate->gate->visited = level;
    cur_size = cur_size2;
    scratch.stack.push_back(agate);

    if (cur_size > max_size) {
      scratch.longest_seq = scratch.stack;
      max_size = cur_size;
    }
  }

  template <typename Scratch, typename GateA>
  static void Pop(unsigned& cur_size, Scratch& scratch, GateA* agate) {
    agate->gate->visited = kFirst;
    cur_size -= agate->qubits.size();
    scratch.stack.pop_back();
  }

  template <typename GateF, typename GateA>
  static void GetNextAvailableGates(unsigned max_fused_size, unsigned cur_size,
                                    const GateF& pgate1, const GateF* pgate2,
                                    std::vector<GateA>& scratch,
                                    std::vector<GateA*>& next_gates) {
    next_gates.resize(0);

    for (auto link : pgate1.links) {
      if (link->next == nullptr) continue;

      auto ngate = link->next->val;

      if (ngate->visited > kFirst) continue;

      GateA next = {ngate, {}, {}};
      next.qubits.reserve(8);
      next.links.reserve(8);

      GetAddedQubits(pgate1, pgate2, *ngate, next);

      if (cur_size + next.qubits.size() > max_fused_size) continue;

      scratch.push_back(std::move(next));
      next_gates.push_back(&scratch.back());
    }
  }

  template <typename GateF, typename GateA>
  static bool GetPrevAvailableGates(unsigned max_fused_size,
                                    unsigned cur_size, unsigned level,
                                    const GateF& ngate1, const GateF* ngate2,
                                    std::vector<GateA>& scratch,
                                    std::vector<GateA*>& prev_gates) {
    prev_gates.resize(0);

    for (auto link : ngate1.links) {
      if (link->prev == nullptr) continue;

      auto pgate = link->prev->val;

      if (pgate->visited == kFinal || pgate->visited == level) continue;

      if (pgate->visited > kFirst) {
        prev_gates.resize(0);
        return false;
      }

      GateA prev = {pgate, {}, {}};
      prev.qubits.reserve(8);
      prev.links.reserve(8);

      GetAddedQubits(ngate1, ngate2, *pgate, prev);

      bool all_prev_visited = true;

      for (auto link : pgate->links) {
        if (link->prev == nullptr) continue;

        if (link->prev->val->visited <= kUnfusible) {
          all_prev_visited = false;
          break;
        }
      }

      if (!all_prev_visited) {
        prev_gates.resize(0);
        return false;
      }

      if (cur_size + prev.qubits.size() > max_fused_size) continue;

      if (all_prev_visited) {
        scratch.push_back(std::move(prev));
        prev_gates.push_back(&scratch.back());
      }
    }

    return true;
  }

  template <typename GateF, typename GateA>
  static void GetAddedQubits(const GateF& fgate0, const GateF* fgate1,
                             const GateF& fgate2, GateA& added) {
    for (std::size_t i = 0; i < fgate2.qubits.size(); ++i) {
      unsigned q2 = fgate2.qubits[i];

      if (std::find(fgate0.qubits.begin(), fgate0.qubits.end(), q2)
          != fgate0.qubits.end()) continue;

      if (fgate1 != nullptr
          && std::find(fgate1->qubits.begin(), fgate1->qubits.end(), q2)
            != fgate1->qubits.end()) continue;

      added.qubits.push_back(q2);
      added.links.push_back(fgate2.links[i]);
    }
  }

  // Fuse smaller gates with fgate back in gate time.
  template <typename Parent, typename PGate>
  static void FusePrev(unsigned pass, GateF<Parent, PGate>& fgate) {
    using Link = Link<Parent, PGate>;

    std::vector<PGate> gates;
    gates.reserve(fgate.gates.capacity());

    auto neighbor = [](const Link* link) -> const Link* {
      return link->prev;
    };

    FusePrevOrNext<std::greater<unsigned>>(pass, neighbor, fgate, gates);

    for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
      fgate.gates.push_back(*it);
    }
  }

  // Fuse smaller gates with fgate forward in gate time.
  template <typename Parent, typename PGate>
  static void FuseNext(unsigned pass, GateF<Parent, PGate>& fgate) {
    using Link = Link<Parent, PGate>;

    auto neighbor = [](const Link* link) -> const Link* {
      return link->next;
    };

    FusePrevOrNext<std::less<unsigned>>(pass, neighbor, fgate, fgate.gates);
  }

  template <typename R, typename Neighbor, typename Parent, typename PGate>
  static void FusePrevOrNext(unsigned pass, Neighbor neighb,
                             GateF<Parent, PGate>& fgate,
                             std::vector<PGate>& gates) {
    using Link = Link<Parent, PGate>;
    using Gate = qsim::Gate<typename GateF<Parent, PGate>::fp_type>;

    uint64_t bad_mask = 0;
    auto links = fgate.links;

    bool may_have_gates_to_fuse = true;

    while (may_have_gates_to_fuse) {
      may_have_gates_to_fuse = false;

      std::sort(links.begin(), links.end(),
                [&neighb](const Link* l, const Link* r) -> bool {
                  auto ln = neighb(l);
                  auto rn = neighb(r);

                  if (ln != nullptr && rn != nullptr) {
                    return R()(OpTime(*ln->val->parent),
                               OpTime(*rn->val->parent));
                  } else {
                    // nullptrs are larger than everything else and
                    // equivalent among each other.
                    return ln != nullptr;
                  }
                });

      for (auto link : links) {
        auto n = neighb(link);

        if (n == nullptr) continue;

        auto g = n->val;

        if (!QubitsAreIn(fgate.mask, g->mask) || (g->mask & bad_mask) != 0
            || g->visited > pass) {
          bad_mask |= g->mask;
        } else {
          g->visited = pass == 0 ? kFirst : kFinal;

          if (pass == 0) {
            // g->parent must hold the type Gate here.
            gates.push_back(OpGetAlternative<Gate>(*g->parent));
          } else {
            for (auto gate : g->gates) {
              gates.push_back(gate);
            }
          }

          for (auto link : g->links) {
            LinkManager<Parent, PGate>::Delete(link);
          }

          may_have_gates_to_fuse = true;
          break;
        }
      }
    }
  }

  static bool QubitsAreIn(uint64_t mask0, uint64_t mask) {
    return ((mask0 | mask) ^ mask0) == 0;
  }

  template <typename OperationF>
  static void PrintStat(unsigned verbosity, const Stat& stat,
                        const std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using ControlledGate = qsim::ControlledGate<fp_type>;

    if (verbosity < 3) return;

    if (stat.num_unfusible_gates > 0) {
      IO::messagef("%u unfusible gates\n", stat.num_unfusible_gates);
    }

    if (stat.num_measurements > 0) {
      IO::messagef("%u measurement gates", stat.num_measurements);
      if (stat.num_fused_measurements == stat.num_measurements) {
        IO::messagef("\n");
      } else {
        IO::messagef(" are fused into %u gates\n", stat.num_fused_measurements);
      }
    }

    bool first = true;
    for (unsigned i = 1; i < stat.num_gates.size(); ++i) {
      if (stat.num_gates[i] > 0) {
        if (first) {
          first = false;
        } else {
          IO::messagef(", ");
        }
        IO::messagef("%u %u-qubit", stat.num_gates[i], i);
      }
    }

    IO::messagef(" gates are fused into %u gates\n", stat.num_fused_gates);

    if (verbosity < 5) return;

    IO::messagef("fused gate qubits:\n");
    for (const auto& op : fused_ops) {
      const auto& bop = OpBaseOperation(op);
      IO::messagef("%6u  ", bop.time);

      if (OpGetAlternative<Measurement>(op)) {
        IO::messagef("m");
      } else if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
        IO::messagef("c");
        for (auto q : pg->controlled_by) {
          IO::messagef("%3u", q);
        }
        IO::messagef("  t");
      } else {
        IO::messagef(" ");
      }

      for (auto q : bop.qubits) {
        IO::messagef("%3u", q);
      }
      IO::messagef("\n");
    }
  }

  template <typename Operation, typename GatesLat>
  static bool ValidateOp(const Operation& op, unsigned max_qubit1,
                         const GatesLat& gates_lat) {
    using ControlledGate = qsim::ControlledGate<OpFpType<Operation>>;

    const auto& bop = OpBaseOperation(op);

    for (unsigned q : bop.qubits) {
      if (q >= max_qubit1) {
        IO::errorf("fuser: gate qubit %u is out of range "
                   "(should be smaller than %u).\n", q, max_qubit1);
        return false;
      }
      if (gates_lat[q] != nullptr
          && bop.time <= OpTime(*gates_lat[q]->val->parent)) {
        IO::errorf("fuser: gate at time %u is out of time order.\n", time);
        return false;
      }
    }

    if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
      for (unsigned q : pg->controlled_by) {
        if (q >= max_qubit1) {
          IO::errorf("fuser: gate qubit %u is out of range "
                     "(should be smaller than %u).\n", q, max_qubit1);
          return false;
        }
        if (gates_lat[q] != nullptr
            && bop.time <= OpTime(*gates_lat[q]->val->parent)) {
          IO::errorf("fuser: gate at time %u is out of time order.\n", time);
          return false;
        }
      }
    }

    return true;
  }
};

}  // namespace qsim

#endif  // FUSER_MQUBIT_H_
