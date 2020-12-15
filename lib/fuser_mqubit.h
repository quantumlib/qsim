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

#include "gate.h"
#include "fuser.h"

namespace qsim {

/**
 * Multi-qubit gate fuser.
 * Measurement gates with equal times are fused together.
 * User-defined controlled gates (controlled_by.size() > 0) are not fused.
 * The template parameter Gate can be Gate type or a pointer to Gate type.
 */
template <typename IO, typename Gate>
class MultiQubitGateFuser {
 private:
  using RGate = typename std::remove_pointer<Gate>::type;

  static const RGate& GateToConstRef(const RGate& gate) {
    return gate;
  }

  static const RGate& GateToConstRef(const RGate* gate) {
    return *gate;
  }

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

  struct GateF;

  using LinkManager = LinkManagerT<GateF*>;
  using Link = typename LinkManager::Link;

  // Intermediate representation of a fused gate.
  struct GateF {
    const RGate* parent;
    std::vector<unsigned> qubits;
    std::vector<const RGate*> gates;  // Gates that get fused to this gate.
    std::vector<Link*> links;         // Gate "lattice" links.
    uint64_t mask;                    // Qubit mask.
    unsigned visited;
  };

  // Possible values for visited in GateF.
  // Note that MakeGateSequence assignes values from kSecond to the number of
  // gates in the sequence plus one, see below.
  enum Visited {
    kZero = 0,             // Start value for "normal" gates.
    kFirst = 1,            // Value after the first pass for partially fused
                           // "normal" gates.
    kSecond = 2,           // Start value to assign values in MakeGateSequence.
    kCompress = 99999997,  // Used to compress links.
    kMeaCnt = 99999998,    // Start value for controlled or measurement gates.
    kFinal = 99999999,     // Value after the second pass for fused "normal"
                           // gates or for controlled and measurement gates.
  };

  struct Stat {
    unsigned num_mea_gates = 0;
    unsigned num_fused_mea_gates = 0;
    unsigned num_fused_gates = 0;
    unsigned num_controlled_gates = 0;
    std::vector<unsigned> num_gates;
  };

  // Gate that is added to a sequence of gates to fuse together.
  struct GateA {
    GateF* gate;
    std::vector<unsigned> qubits;  // Added qubits.
    std::vector<Link*> links;      // Added lattice links.
  };

  struct Scratch {
    std::vector<GateA> data;
    std::vector<GateA*> prev1;
    std::vector<GateA*> prev2;
    std::vector<GateA*> next1;
    std::vector<GateA*> next2;
    std::vector<GateA*> longest_seq;
    std::vector<GateA*> stack;
    std::vector<GateF*> gates;
    unsigned count = 0;
  };

 public:
  using GateFused = qsim::GateFused<RGate>;

  /**
   * User-specified parameters for gate fusion.
   */
  struct Parameter {
    /**
     * Maximum number of qubits in a fused gate. It can take values from 2 to
     * 6 (0 and 1 are equivalent to 2). It is not recommended to use 5 or 6 as
     * that might degrade performance for not very fast machines.
     */
    unsigned max_fused_size = 2;
    unsigned verbosity = 0;
  };

  /**
   * Stores ordered sets of gates that can be applied together. Note that
   * gates fused with this method are not multiplied together until
   * ApplyFusedGate is called on the output. To respect specific time
   * boundaries while fusing gates, use the other version of this method below.
   * @param param Options for gate fusion.
   * @param num_qubits The number of qubits acted on by 'gates'.
   * @param gates The gates (or pointers to the gates) to be fused.
   *   Gate times should be ordered.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(const Parameter& param,
                                          unsigned num_qubits,
                                          const std::vector<Gate>& gates) {
    return FuseGates(param, num_qubits, gates.cbegin(), gates.cend(), {});
  }

  /**
   * Stores ordered sets of gates that can be applied together. Note that
   * gates fused with this method are not multiplied together until
   * ApplyFusedGate is called on the output.
   * @param param Options for gate fusion.
   * @param num_qubits The number of qubits acted on by 'gates'.
   * @param gates The gates (or pointers to the gates) to be fused.
   *   Gate times should be ordered.
   * @param times_to_split_at Ordered list of time steps at which to separate
   *   fused gates. Each element of the output will contain gates from a single
   *   'window' in this list.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param,
      unsigned num_qubits, const std::vector<Gate>& gates,
      const std::vector<unsigned>& times_to_split_at) {
    return FuseGates(param, num_qubits, gates.cbegin(), gates.cend(),
                     times_to_split_at);
  }

  /**
   * Stores ordered sets of gates that can be applied together. Note that
   * gates fused with this method are not multiplied together until
   * ApplyFusedGate is called on the output. To respect specific time
   * boundaries while fusing gates, use the other version of this method below.
   * @param param Options for gate fusion.
   * @param num_qubits The number of qubits acted on by gates.
   * @param gfirst, glast The iterator range [gfirst, glast) to fuse gates
   *   (or pointers to gates) in. Gate times should be ordered.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param, unsigned num_qubits,
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast) {
    return FuseGates(param, num_qubits, gfirst, glast, {});
  }

  /**
   * Stores ordered sets of gates that can be applied together. Note that
   * gates fused with this method are not multiplied together until
   * ApplyFusedGate is called on the output.
   * @param param Options for gate fusion.
   * @param num_qubits The number of qubits acted on by gates.
   * @param gfirst, glast The iterator range [gfirst, glast) to fuse gates
   *   (or pointers to gates) in. Gate times should be ordered.
   * @param times_to_split_at Ordered list of time steps at which to separate
   *   fused gates. Each element of the output will contain gates from a single
   *   'window' in this list.
   * @return A vector of fused gate objects. Each element is a set of gates
   *   acting on a specific pair of qubits which can be applied as a group.
   */
  static std::vector<GateFused> FuseGates(
      const Parameter& param, unsigned num_qubits,
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      const std::vector<unsigned>& times_to_split_at) {
    std::vector<GateFused> fused_gates;

    if (gfirst >= glast) return fused_gates;

    std::size_t num_gates = glast - gfirst;

    fused_gates.reserve(num_gates);

    // Merge with measurement gate times to separate fused gates at.
    auto epochs = MergeWithMeasurementTimes(gfirst, glast, times_to_split_at);

    LinkManager link_manager(num_qubits * num_gates);

    // Auxillary data structures.
    // Sequence of intermediate fused gates.
    std::vector<GateF> gates_seq;
    // Gate "lattice".
    std::vector<Link*> gates_lat;
    // Sequences of intermediate fused gates ordered by gate size.
    std::vector<std::vector<GateF*>> fgates(num_qubits + 1);

    gates_seq.reserve(num_gates);
    gates_lat.reserve(num_qubits);

    Scratch scratch;

    scratch.data.reserve(1024);
    scratch.prev1.reserve(32);
    scratch.prev2.reserve(32);
    scratch.next1.reserve(32);
    scratch.next2.reserve(32);
    scratch.longest_seq.reserve(8);
    scratch.stack.reserve(8);

    Stat stat;
    stat.num_gates.resize(num_qubits + 1, 0);

    unsigned max_fused_size = std::min(unsigned{6}, param.max_fused_size);
    max_fused_size = std::min(max_fused_size, num_qubits);

    auto gate_it = gfirst;

    // Iterate over epochs.
    for (std::size_t l = 0; l < epochs.size(); ++l) {
      gates_seq.resize(0);
      gates_lat.resize(0);
      gates_lat.resize(num_qubits, nullptr);

      for (unsigned i = 0; i <= num_qubits; ++i) {
        fgates[i].resize(0);
      }

      uint64_t max_gate_size = 0;
      GateF* last_mea_gate = nullptr;

      auto prev_time = GateToConstRef(*gate_it).time;

      // Iterate over input gates.
      for (; gate_it < glast; ++gate_it) {
        const auto& gate = GateToConstRef(*gate_it);

        if (gate.time > epochs[l]) break;

        if (gate.time < prev_time) {
          // This function assumes that gate times are ordered.
          // Just stop if this is not the case.
          IO::errorf("gate times should be ordered.\n");
          fused_gates.resize(0);
          return fused_gates;
        }

        prev_time = gate.time;

        // Fill in auxillary data structures.

        if (gate.kind == gate::kMeasurement) {
          // Measurement gate.

          if (last_mea_gate == nullptr
              || last_mea_gate->parent->time != gate.time) {
            gates_seq.push_back({&gate, {}, {}, {}, 0, kMeaCnt});
            last_mea_gate = &gates_seq.back();

            last_mea_gate->qubits.reserve(num_qubits);
            last_mea_gate->links.reserve(num_qubits);

            ++stat.num_fused_mea_gates;
          }

          for (auto q : gate.qubits) {
            last_mea_gate->qubits.push_back(q);
            last_mea_gate->mask |= uint64_t{1} << q;
            gates_lat[q] = link_manager.AddBack(last_mea_gate, gates_lat[q]);
            last_mea_gate->links.push_back(gates_lat[q]);
          }

          last_mea_gate->gates.push_back(&gate);

          ++stat.num_mea_gates;
        } else {
          gates_seq.push_back({&gate, {}, {}, {}, 0, kZero});
          auto& fgate = gates_seq.back();

          if (gate.controlled_by.size() == 0) {
            if (max_gate_size < gate.qubits.size()) {
              max_gate_size = gate.qubits.size();
            }

            unsigned num_gate_qubits = gate.qubits.size();
            unsigned size = std::max(max_fused_size, num_gate_qubits);

            fgate.qubits.reserve(size);
            fgate.links.reserve(size);
            fgate.gates.reserve(4 * size);
            fgate.links.reserve(size);

            if (fgates[num_gate_qubits].empty()) {
              fgates[num_gate_qubits].reserve(num_gates);
            }
            fgates[num_gate_qubits].push_back(&fgate);

            ++stat.num_gates[num_gate_qubits];
          } else {
            // Controlled gate.
            // Controlled gates are not fused with other gates.

            uint64_t size = gate.qubits.size() + gate.controlled_by.size();

            fgate.qubits.reserve(gate.qubits.size());
            fgate.links.reserve(size);

            fgate.visited = kMeaCnt;
            fgate.gates.push_back(&gate);

            ++stat.num_controlled_gates;
          }

          for (auto q : gate.qubits) {
            fgate.qubits.push_back(q);
            fgate.mask |= uint64_t{1} << q;
            gates_lat[q] = link_manager.AddBack(&fgate, gates_lat[q]);
            fgate.links.push_back(gates_lat[q]);
          }

          for (auto q : gate.controlled_by) {
            fgate.mask |= uint64_t{1} << q;
            gates_lat[q] = link_manager.AddBack(&fgate, gates_lat[q]);
            fgate.links.push_back(gates_lat[q]);
          }
        }
      }

      // Fuse large gates with smaller gates.
      FuseGates(max_gate_size, fgates);

      if (max_fused_size > 2) {
        FuseGateSequences(
            max_fused_size, num_qubits, scratch, gates_seq, stat, fused_gates);
      } else {
        for (auto& fgate : gates_seq) {
          if (fgate.gates.size() > 0) {
            // Assume fgate.qubits (gate.qubits) are sorted.
            fused_gates.push_back({fgate.parent->kind, fgate.parent->time,
                                   std::move(fgate.qubits), fgate.parent,
                                   std::move(fgate.gates)});

            if (fgate.visited != kMeaCnt) {
              ++stat.num_fused_gates;
            }
          }
        }
      }
    }

    PrintStat(param.verbosity, stat, fused_gates);

    return fused_gates;
  }

 private:
  static std::vector<unsigned> MergeWithMeasurementTimes(
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      const std::vector<unsigned>& times) {
    std::vector<unsigned> epochs;
    epochs.reserve(glast - gfirst + times.size());

    std::size_t last = 0;

    for (auto gate_it = gfirst; gate_it < glast; ++gate_it) {
      const auto& gate = GateToConstRef(*gate_it);

      if (gate.kind == gate::kMeasurement
          && (epochs.size() == 0 || epochs.back() < gate.time)) {
        epochs.push_back(gate.time);
      }

      if (last < times.size() && gate.time > times[last]) {
        while (last < times.size() && times[last] <= gate.time) {
          unsigned prev = times[last++];
          epochs.push_back(prev);
          while (last < times.size() && times[last] <= prev) ++last;
        }
      }
    }

    const auto& back = *(glast - 1);

    if (epochs.size() == 0 || epochs.back() < GateToConstRef(back).time) {
      epochs.push_back(GateToConstRef(back).time);
    }

    return epochs;
  }

  // Fuse large gates with smaller gates.
  static void FuseGates(uint64_t max_gate_size,
                        std::vector<std::vector<GateF*>>& fgates) {
    // Traverse gates in order of decreasing size.
    for (uint64_t i = 0; i < max_gate_size; ++i) {
      std::size_t pos = 0;

      for (auto fgate : fgates[max_gate_size - i]) {
        if (fgate->visited > kZero) continue;

        fgates[max_gate_size - i][pos++] = fgate;

        fgate->visited = kFirst;

        FusePrev(0, *fgate);
        fgate->gates.push_back(fgate->parent);
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
  static void FuseGateSequences(unsigned max_fused_size,
                                unsigned num_qubits, Scratch& scratch,
                                std::vector<GateF>& gates_seq, Stat& stat,
                                std::vector<GateFused>& fused_gates) {
    unsigned prev_time = 0;

    std::vector<GateF*> orphaned_gates;
    orphaned_gates.reserve(num_qubits);

    for (auto& fgate : gates_seq) {
      if (prev_time != fgate.parent->time) {
        if (orphaned_gates.size() > 0) {
          FuseOrphanedGates(max_fused_size, stat, orphaned_gates, fused_gates);
          orphaned_gates.resize(0);
        }

        prev_time = fgate.parent->time;
      }

      if (fgate.visited == kFinal || fgate.gates.size() == 0) continue;

      if (fgate.visited == kMeaCnt || fgate.qubits.size() >= max_fused_size
          || fgate.parent->unfusible) {
        if (fgate.visited != kMeaCnt) {
           ++stat.num_fused_gates;
        }

        fgate.visited = kFinal;

        fused_gates.push_back({fgate.parent->kind, fgate.parent->time,
                               std::move(fgate.qubits), fgate.parent,
                               std::move(fgate.gates)});

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

          fused_gates.push_back({fgate->parent->kind, fgate->parent->time,
                                 std::move(fgate->qubits), fgate->parent,
                                 std::move(fgate->gates)});

          ++stat.num_fused_gates;
        }
      }
    }

    if (orphaned_gates.size() > 0) {
      FuseOrphanedGates(max_fused_size, stat, orphaned_gates, fused_gates);
    }
  }

  static void FuseOrphanedGates(unsigned max_fused_size, Stat& stat,
                                std::vector<GateF*>& orphaned_gates,
                                std::vector<GateFused>& fused_gates) {
    unsigned count = 0;

    for (std::size_t i = 0; i < orphaned_gates.size(); ++i) {
      auto ogate1 = orphaned_gates[i];

      if (ogate1->visited == kFinal) continue;

      ogate1->visited = kFinal;

      for (std::size_t j = i + 1; j < orphaned_gates.size(); ++j) {
        auto ogate2 = orphaned_gates[j];

        if (ogate2->visited == kFinal) continue;

        ++count;

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

      fused_gates.push_back({ogate1->parent->kind, ogate1->parent->time,
                             std::move(ogate1->qubits), ogate1->parent,
                             std::move(ogate1->gates)});

      ++stat.num_fused_gates;
    }
  }

  static void MakeGateSequence(
      unsigned max_fused_size, Scratch& scratch, GateF& fgate) {
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
        LinkManager::Delete(link->next);
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

  static void AddGatesFromNext(std::vector<const RGate*>& gates, GateF& fgate) {
    for (auto gate : gates) {
      fgate.gates.push_back(gate);
    }
  }

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

  static void FindLongestGateSequence(
      unsigned max_fused_size, unsigned level, Scratch& scratch, GateF& fgate) {
    scratch.data.push_back({&fgate, {}, {}});

    scratch.longest_seq.resize(0);
    scratch.longest_seq.push_back(&scratch.data.back());

    scratch.stack.resize(0);
    scratch.stack.push_back(&scratch.data.back());

    unsigned cur_size = fgate.qubits.size();
    fgate.visited = level;

    unsigned max_size = cur_size;

    GetNextAvailableGates(max_fused_size, cur_size, fgate, nullptr,
                          scratch.data, scratch.next1);

    for (auto n1 : scratch.next1) {
      unsigned cur_size2 = cur_size + n1->qubits.size();
      if (cur_size2 > max_fused_size) continue;

      bool feasible = GetPrevAvailableGates(max_fused_size, cur_size,
                                            level, *n1->gate, nullptr,
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

  static void Pop(unsigned& cur_size, Scratch& scratch, GateA* agate) {
    agate->gate->visited = kFirst;
    cur_size -= agate->qubits.size();
    scratch.stack.pop_back();
  }

  static void GetNextAvailableGates(unsigned max_fused_size, unsigned cur_size,
                                    const GateF& pgate1, const GateF* pgate2,
                                    std::vector<GateA>& scratch,
                                    std::vector<GateA*>& next_gates) {
    next_gates.resize(0);

    for (auto link : pgate1.links) {
      if (link->next == nullptr) continue;

      auto ngate = link->next->val;

      if (ngate->visited > kFirst || ngate->parent->unfusible) continue;

      GateA next = {ngate, {}, {}};
      next.qubits.reserve(8);
      next.links.reserve(8);

      GetAddedQubits(pgate1, pgate2, *ngate, next);

      if (cur_size + next.qubits.size() > max_fused_size) continue;

      scratch.push_back(std::move(next));
      next_gates.push_back(&scratch.back());
    }
  }

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

      if (pgate->visited > kFirst || pgate->parent->unfusible) {
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

        if (link->prev->val->visited <= kMeaCnt) {
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
  static void FusePrev(unsigned pass, GateF& fgate) {
    std::vector<const RGate*> gates;
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
  static void FuseNext(unsigned pass, GateF& fgate) {
    auto neighbor = [](const Link* link) -> const Link* {
      return link->next;
    };

    FusePrevOrNext<std::less<unsigned>>(pass, neighbor, fgate, fgate.gates);
  }

  template <typename R, typename Neighbor>
  static void FusePrevOrNext(unsigned pass, Neighbor neighb, GateF& fgate,
                             std::vector<const RGate*>& gates) {
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
                    return R()(ln->val->parent->time, rn->val->parent->time);
                  } else {
                    return ln != nullptr || rn == nullptr;
                  }
                });

      for (auto link : links) {
        auto n = neighb(link);

        if (n == nullptr) continue;

        auto g = n->val;

        if (!QubitsAreIn(fgate.mask, g->mask) || (g->mask & bad_mask) != 0
            || g->visited > pass || g->parent->unfusible) {
          bad_mask |= g->mask;
        } else {
          g->visited = pass == 0 ? kFirst : kFinal;

          if (pass == 0) {
            gates.push_back(g->parent);
          } else {
            for (auto gate : g->gates) {
              gates.push_back(gate);
            }
          }

          for (auto link : g->links) {
            LinkManager::Delete(link);
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

  static void PrintStat(unsigned verbosity, const Stat& stat,
                        const std::vector<GateFused>& fused_gates) {
    if (verbosity == 0) return;

    if (stat.num_controlled_gates > 0) {
      IO::messagef("%lu controlled gates\n", stat.num_controlled_gates);
    }

    if (stat.num_mea_gates > 0) {
      IO::messagef("%lu measurement gates", stat.num_mea_gates);
      if (stat.num_fused_mea_gates == stat.num_mea_gates) {
        IO::messagef("\n");
      } else {
        IO::messagef(" are fused into %lu gates\n", stat.num_fused_mea_gates);
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

    IO::messagef(" gates are fused into %lu gates\n", stat.num_fused_gates);

    if (verbosity == 1) return;

    IO::messagef("fused gate qubits:\n");
    for (const auto g : fused_gates) {
      IO::messagef("%6u  ", g.parent->time);
      if (g.parent->kind == gate::kMeasurement) {
        IO::messagef("m");
      } else if (g.parent->controlled_by.size() > 0) {
        IO::messagef("c");
        for (auto q : g.parent->controlled_by) {
          IO::messagef("%3u", q);
        }
        IO::messagef("  t");
      } else {
        IO::messagef(" ");
      }

      for (auto q : g.qubits) {
        IO::messagef("%3u", q);
      }
      IO::messagef("\n");
    }
  }
};

}  // namespace qsim

#endif  // FUSER_MQUBIT_H_
