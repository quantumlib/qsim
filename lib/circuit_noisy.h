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

#ifndef CIRCUIT_NOISY_H_
#define CIRCUIT_NOISY_H_

#include <vector>

#include "operation.h"

namespace qsim {

/**
 * Makes a noisy circuit from the clean circuit.
 * Channels are added after each qubit of each gate of the clean cicuit.
 * Roughly equivalent to cirq.Circuit.with_noise.
 * @param num_qubits The number of circuit qubits.
 * @param gbeg, gend The iterator range [gbeg, gend) of circuit gates.
 * @param A channel factory to construct channels.
 * @return The output noisy circuit.
 */
template <typename FP, typename ChannelFactory>
inline Circuit<Operation<FP>> MakeNoisy(
    unsigned num_qubits,
    typename std::vector<Operation<FP>>::const_iterator obeg,
    typename std::vector<Operation<FP>>::const_iterator oend,
    const ChannelFactory& channel_factory) {
  Circuit<Operation<FP>> ncircuit;

  ncircuit.num_qubits = num_qubits;
  ncircuit.ops.reserve(4 * std::size_t(oend - obeg));

  for (auto it = obeg; it != oend; ++it) {
    const auto& op = *it;

    const auto& bop = OpBaseOperation(op);

    ncircuit.ops.push_back(op);
    OpBaseOperation(ncircuit.ops.back()).time = 2 * bop.time;

    for (auto q : bop.qubits) {
      ncircuit.ops.push_back(channel_factory.Create(2 * bop.time + 1, q));
    }

    if (const auto* pg = OpGetAlternative<ControlledGate<FP>>(op)) {
      for (auto q : pg->controlled_by) {
        ncircuit.ops.push_back(channel_factory.Create(2 * bop.time + 1, q));
      }
    }
  }

  return ncircuit;
}

/**
 * Makes a noisy circuit from the clean circuit.
 * Channels are added after each qubit of each gate of the clean cicuit.
 * Roughly equivalent to cirq.Circuit.with_noise.
 * @param num_qubits The number of circuit qubits.
 * @param gates The circuit gates.
 * @param A channel factory to construct channels.
 * @return The output noisy circuit.
 */
template <typename FP, typename ChannelFactory>
inline Circuit<Operation<FP>> MakeNoisy(
    unsigned num_qubits, const std::vector<Operation<FP>>& ops,
    const ChannelFactory& channel_factory) {
  return MakeNoisy<FP>(num_qubits, ops.begin(), ops.end(), channel_factory);
}

/**
 * Makes a noisy circuit from the clean circuit.
 * Channels are added after each qubit of each gate of the clean cicuit.
 * Roughly equivalent to cirq.Circuit.with_noise.
 * @param circuit The input cicuit.
 * @param A channel factory to construct channels.
 * @return The output noisy circuit.
 */
template <typename FP, typename ChannelFactory>
inline Circuit<Operation<FP>> MakeNoisy(
    const Circuit<Operation<FP>>& circuit,
    const ChannelFactory& channel_factory) {
  return MakeNoisy<FP, ChannelFactory>(circuit.num_qubits, circuit.ops.begin(),
                       circuit.ops.end(), channel_factory);
}

}  // namespace qsim

#endif  // CIRCUIT_NOISY_H_
