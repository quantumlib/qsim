# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq

import qsimcirq

# Create a Bell state, |00) + |11)
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))

# Constructs a noise model that adds depolarizing noise after each gate.
noise = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p=0.05))

# Use the noise model to create a noisy circuit.
noisy_circuit = cirq.Circuit(noise.noisy_moments(circuit, system_qubits=[q0, q1]))

sim = qsimcirq.QSimSimulator()
result = sim.run(noisy_circuit, repetitions=1000)
# Outputs a histogram dict of result:count pairs.
# Expected result is a bunch of 0s and 3s, with fewer 1s and 2s.
# (For comparison, the noiseless circuit will only have 0s and 3s)
print(result.histogram(key="m"))
