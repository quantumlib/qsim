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

#ifndef CHANNEL_H_
#define CHANNEL_H_

#include <set>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace qsim {

/**
 * Kraus operator.
 */
template <typename Gate>
struct KrausOperator {
  using fp_type = typename Gate::fp_type;

  enum Kind {
    kNormal = 0,
    kMeasurement = gate::kMeasurement,
  };

  /**
   * Kraus operator type;
   */
  Kind kind;

  /**
   * If true, the Kraus operator is a unitary operator times a constant.
   */
  bool unitary;

  /**
   * Lower bound on Kraus operator probability.
   */
  double prob;

  /**
   * Sequence of operations that represent the Kraus operator. This can be just
   * one operation.
   */
  std::vector<Gate> ops;

  /**
   * Product of K^\dagger and K. This can be empty if unitary = true.
   */
  Matrix<fp_type> kd_k;

  /**
   * Qubits kd_k acts on. This can be empty if unitary = true.
   */
  std::vector<unsigned> qubits;

  /**
   * Calculates the product of "K^\dagger K". Sets qubits "K^\dagger K" acts on.
   */
  void CalculateKdKMatrix() {
    if (ops.size() == 1) {
      kd_k = ops[0].matrix;
      MatrixDaggerMultiply(ops[0].qubits.size(), ops[0].matrix, kd_k);
      qubits = ops[0].qubits;
    } else if (ops.size() > 1) {
      std::set<unsigned> qubit_map;

      for (const auto& op : ops) {
        for (unsigned q : op.qubits) {
          qubit_map.insert(q);
        }
      }

      unsigned num_qubits = qubit_map.size();

      qubits.resize(0);
      qubits.reserve(num_qubits);

      for (auto it = qubit_map.begin(); it != qubit_map.end(); ++it) {
        qubits.push_back(*it);
      }

      MatrixIdentity(unsigned{1} << num_qubits, kd_k);

      for (const auto& op : ops) {
        if (op.qubits.size() == num_qubits) {
          MatrixMultiply(num_qubits, op.matrix, kd_k);
        } else {
          unsigned mask = 0;

          for (auto q : op.qubits) {
            for (unsigned i = 0; i < num_qubits; ++i) {
              if (q == qubits[i]) {
                mask |= unsigned{1} << i;
                break;
              }
            }
          }

          MatrixMultiply(mask, op.qubits.size(), op.matrix, num_qubits, kd_k);
        }
      }

      auto m = kd_k;
      MatrixDaggerMultiply(num_qubits, m, kd_k);
    }
  }
};

/**
 * Quantum channel.
 */
template <typename Gate>
using Channel = std::vector<KrausOperator<Gate>>;

/**
 * Makes a channel from the gate.
 * @param time The time to place the channel at.
 * @param gate The input gate.
 * @return The output channel.
 */
template <typename Gate>
Channel<Gate> MakeChannelFromGate(unsigned time, const Gate& gate) {
  auto normal = KrausOperator<Gate>::kNormal;
  auto measurement = KrausOperator<Gate>::kMeasurement;

  auto kind = gate.kind == gate::kMeasurement ? measurement : normal;

  Channel<Gate> channel = {{kind, true, 1, {gate}}};
  channel[0].ops[0].time = time;

  return channel;
}

}  // namespace qsim

#endif  // CHANNEL_H_
