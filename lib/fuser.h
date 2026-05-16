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

#ifndef FUSER_H_
#define FUSER_H_

#include <cstdint>
#include <vector>

#include "gate.h"
#include "matrix.h"
#include "operation_base.h"

namespace qsim {

/**
 * Base class for fuser classes with some common functions.
 */
template <typename IO>
class Fuser {
 protected:
  template <typename Operation>
  static const Operation& OperationToConstRef(const Operation& op) {
    return op;
  }

  template <typename Operation>
  static const Operation& OperationToConstRef(const Operation* op) {
    return *op;
  }

  template <typename Operation>
  static std::vector<unsigned> MergeWithMeasurementTimes(
      typename std::vector<Operation>::const_iterator obeg,
      typename std::vector<Operation>::const_iterator oend,
      const std::vector<unsigned>& times) {
    std::vector<unsigned> windows;
    windows.reserve(oend - obeg + times.size());

    std::size_t last = 0;
    unsigned max_time = 0;

    for (auto op_it = obeg; op_it < oend; ++op_it) {
      const auto& op = OperationToConstRef(*op_it);

      unsigned time = OpTime(op);

      if (time > max_time) {
        max_time = time;
      }

      if (windows.size() > 0 && time < windows.back()) {
        IO::errorf("gate crosses the time boundary.\n");
        windows.resize(0);
        return windows;
      }

      if (OpGetAlternative<Measurement>(op)) {
        if (windows.size() == 0 || windows.back() < time) {
          if (!AddBoundary(time, max_time, windows)) {
            windows.resize(0);
            return windows;
          }
        }
      }

      while (last < times.size() && times[last] <= time) {
        unsigned prev = times[last++];
        windows.push_back(prev);
        if (!AddBoundary(prev, max_time, windows)) {
          windows.resize(0);
          return windows;
        }
        while (last < times.size() && times[last] <= prev) ++last;
      }
    }

    if (windows.size() == 0 || windows.back() < max_time) {
      windows.push_back(max_time);
    }

    return windows;
  }

  template <typename GateSeq0, typename Parent, typename OperationF>
  static void FuseZeroQubitGates(const GateSeq0& gate_seq0,
                                 Parent&& parent, std::size_t first,
                                 std::vector<OperationF>& fused_ops) {
    using FusedGate = std::variant_alternative_t<0, OperationF>;
    using fp_type = typename FusedGate::fp_type;
    using Gate = qsim::Gate<fp_type>;

    FusedGate* fuse_to = nullptr;

    for (std::size_t i = first; i < fused_ops.size(); ++i) {
      auto& fop = fused_ops[i];

      fuse_to = OpGetAlternative<FusedGate>(fop);
      if (fuse_to != nullptr) {
        break;
      }
    }

    if (fuse_to != nullptr) {
      // Fuse zero-qubit gates with the first available fused gate.
      for (const auto& g : gate_seq0) {
        fuse_to->gates.push_back(OpGetAlternative<Gate>(*parent(g)));
      }
    } else {
      const auto& gate0 = *OpGetAlternative<Gate>(*parent(gate_seq0[0]));
      FusedGate fgate{gate0.kind, gate0.time, {}, &gate0, {&gate0}, {}};

      for (std::size_t i = 1; i < gate_seq0.size(); ++i) {
        fgate.gates.push_back(OpGetAlternative<Gate>(*parent(gate_seq0[i])));
      }

      fused_ops.push_back(std::move(fgate));
    }
  }

 private:
  static bool AddBoundary(unsigned time, unsigned max_time,
                          std::vector<unsigned>& boundaries) {
    if (max_time > time) {
      IO::errorf("gate crosses the time boundary.\n");
      return false;
    }

    boundaries.push_back(time);
    return true;
  }
};

/**
 * Multiplies component gate matrices of a fused gate.
 * @param gate Fused gate.
 */
template <typename FP>
inline void CalculateFusedMatrix(FusedGate<FP>& gate) {
  MatrixIdentity(unsigned{1} << gate.qubits.size(), gate.matrix);

  for (const auto& pgate : gate.gates) {
    const auto* pg = OpGetAlternative<Gate<FP>>(pgate);
    const auto& pqubits = OpQubits(pgate);
    const auto& pmatrix =
        pg ? pg->matrix : OpGetAlternative<DecomposedGate<FP>>(pgate)->matrix;

    if (pqubits.size() == 0) {
      MatrixScalarMultiply(pmatrix[0], pmatrix[1], gate.matrix);
    } else if (gate.qubits.size() == pqubits.size()) {
      MatrixMultiply(gate.qubits.size(), pmatrix, gate.matrix);
    } else {
      unsigned mask = 0;

      for (auto q : pqubits) {
        for (std::size_t i = 0; i < gate.qubits.size(); ++i) {
          if (q == gate.qubits[i]) {
            mask |= unsigned{1} << i;
            break;
          }
        }
      }

      MatrixMultiply(mask, pqubits.size(), pmatrix,
                     gate.qubits.size(), gate.matrix);
    }
  }
}

}  // namespace qsim

#endif  // FUSER_H_
