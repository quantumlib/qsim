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
#define CIRCUIT_NOISY_H__

#include <vector>

#include "circuit.h"
#include "channel.h"

namespace qsim {

/**
 * Noisy circuit.
 */
template <typename Gate>
using NoisyCircuit = std::vector<Channel<Gate>>;

/**
 * Makes a noisy circuit from the clean circuit.
 * Channels are added after each qubit of each gate of the clean cicuit.
 * Roughly equivalent to cirq.Circuit.with_noise.
 * @param num_qubits The number of circuit qubits.
 * @param gates The circuit gates.
 * @param A channel factory to construct channels.
 * @return The output noisy circuit.
 */
template <typename Gate, typename ChannelFactory>
inline NoisyCircuit<Gate> MakeNoisy(unsigned num_qubits,
                                    const std::vector<Gate>& gates,
                                    const ChannelFactory& channel_factory) {
  NoisyCircuit<Gate> ncircuit;
  ncircuit.reserve(4 * gates.size());

  for (const auto& gate : gates) {
    ncircuit.push_back(MakeChannelFromGate(2 * gate.time, gate));

    for (auto q : gate.qubits) {
      ncircuit.push_back(channel_factory.Create(2 * gate.time + 1, q));
    }

    for (auto q : gate.controlled_by) {
      ncircuit.push_back(channel_factory.Create(2 * gate.time + 1, q));
    }
  }

  return ncircuit;
}

/**
 * Makes a noisy circuit from the clean circuit.
 * Channels are added after each qubit of each gate of the clean cicuit.
 * Roughly equivalent to cirq.Circuit.with_noise.
 * @param num_qubits The number of circuit qubits.
 * @param circuit The input cicuit.
 * @param A channel factory to construct channels.
 * @return The output noisy circuit.
 */
template <typename Gate, typename ChannelFactory>
inline NoisyCircuit<Gate> MakeNoisy(const Circuit<Gate>& circuit,
                                    const ChannelFactory& channel_factory) {
  return MakeNoisy(circuit.num_qubits, circuit.gates, channel_factory);
}

}  // namespace qsim

#endif  // CIRCUIT_NOISY_H_
