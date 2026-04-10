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

#ifndef FUSER_TESTFIXTURE_H_
#define FUSER_TESTFIXTURE_H_

#include <algorithm>
#include <map>
#include <vector>
#include <type_traits>

#include "gtest/gtest.h"

#include "../lib/gate.h"
#include "../lib/operation.h"
#include "../lib/operation_base.h"

namespace qsim {

template <typename Operation, typename Gate>
bool TestGate(const std::vector<Operation>& ops, const Gate& gate,
              std::vector<unsigned>& op_map) {
  unsigned k = (std::size_t(&gate) - std::size_t(ops.data())) / sizeof(ops[0]);

  if (op_map[k] != 0) {
    return false;
  } else {
    op_map[k] = 1;
    return true;
  }
}

template <typename Operation, typename Map>
bool TestMeasurement(const std::vector<Operation>& ops,
                     const Measurement& mea, std::vector<unsigned>& op_map,
                     const Map& mea_qubits, const Map& mea_at_time) {
  std::vector<unsigned> qubits0;
  qubits0.reserve(64);

  if (std::size_t(&mea) >= std::size_t(ops.data())
      && std::size_t(&mea) <= std::size_t(&ops.back())) {
    unsigned k = (std::size_t(&mea) - std::size_t(ops.data())) / sizeof(ops[0]);

    if (op_map[k] != 0) {
      return false;
    } else {
      op_map[k] = 1;
      return true;
    }
  }

  auto it1 = mea_at_time.find(mea.time);
  if (it1 == mea_at_time.end()) {
    return false;
  }

  const auto& indices = it1->second;
  for (unsigned i : indices) {
    if (op_map[i] != 0) {
      return false;
    } else {
      op_map[i] = 1;
    }
  }

  std::vector<unsigned> qubits = mea.qubits;
  std::sort(qubits.begin(), qubits.end());

  auto it2 = mea_qubits.find(mea.time);
  return it2 != mea_qubits.end() && qubits == it2->second;
}

template <typename Operation, typename OperationF>
bool TestFusedGates(unsigned num_qubits,
                    const std::vector<Operation>& ops,
                    const std::vector<OperationF>& fused_ops) {
  using Gate = qsim::Gate<float>;
  using ControlledGate = qsim::ControlledGate<float>;
  using DecomposedGate = qsim::DecomposedGate<float>;
  using FusedGate = qsim::FusedGate<float>;

  std::vector<unsigned> times(num_qubits, 0);
  std::vector<unsigned> op_map(ops.size(), 0);

  std::map<unsigned, std::vector<unsigned>> mea_qubits;
  std::map<unsigned, std::vector<unsigned>> mea_at_time;

  for (unsigned i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];

    if (const auto* pg = OpGetAlternative<Measurement>(op)) {
      auto& qubits = mea_qubits[pg->time];
      if (qubits.size() == 0) {
        qubits.reserve(8);
      }

      for (auto q : pg->qubits) {
        qubits.push_back(q);
      }

      auto& indices = mea_at_time[pg->time];
      if (indices.size() == 0) {
        indices.reserve(4);
      }

      indices.push_back(i);
    }
  }

  for (auto it = mea_qubits.begin(); it != mea_qubits.end(); ++it) {
    std::sort(it->second.begin(), it->second.end());
  }

  for (const auto& op : fused_ops) {
    if (const auto* pg = OpGetAlternative<FusedGate>(op)) {
      for (const auto& pv : pg->gates) {
        if (const auto* pg = OpGetAlternative<Gate>(pv)) {
          if (!TestGate(ops, *pg, op_map)) {
            return false;
          }

          for (auto q : pg->qubits) {
            if (pg->time < times[q]) {
              return false;
            }
            times[q] = pg->time;
          }
        } else if (const auto* pg = OpGetAlternative<DecomposedGate>(pv)) {
          if (!TestGate(ops, *pg, op_map)) {
            return false;
          }

          for (auto q : pg->qubits) {
            if (pg->time < times[q]) {
              return false;
            }
            times[q] = pg->time;
          }
        }
      }
    } else if (const auto* pg = OpGetAlternative<Measurement>(op)) {
      // Measurements can be fused or unfused.

      if (!TestMeasurement(ops, *pg, op_map, mea_qubits, mea_at_time)) {
        return false;
      }
    } else if (const auto* pg = OpGetAlternative<ControlledGate>(op)) {
      if (!TestGate(ops, *pg, op_map)) {
        return false;
      }

      for (auto q : pg->qubits) {
        if (pg->time < times[q]) {
          return false;
        }
        times[q] = pg->time;
      }

      for (auto q : pg->controlled_by) {
        if (pg->time < times[q]) {
          return false;
        }
        times[q] = pg->time;
      }
    }
  }

  // Test if all gates are present only once.
  for (auto m : op_map) {
    if (m != 1) {
      return false;
    }
  }

  return true;
}

}  // namespace qsim

#endif  // FUSER_TESTFIXTURE_H_
