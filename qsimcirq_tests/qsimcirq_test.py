# Copyright 2019 Google LLC. All Rights Reserved.
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

import unittest
import cirq
import qsimcirq


class MainTest(unittest.TestCase):

  def test_cirq_qsim_simulate(self):
    # Pick qubits.
    a, b, c, d = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0)
    ]

    # Create a circuit
    cirq_circuit = cirq.Circuit(
        cirq.X(a)**0.5,  # Square root of X.
        cirq.Y(b)**0.5,  # Square root of Y.
        cirq.Z(c),  # Z.
        cirq.CZ(a, d)  # ControlZ.
    )

    qsim_circuit = qsimcirq.QSimCircuit(cirq_circuit)

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.compute_amplitudes(
        qsim_circuit, bitstrings=[0b0100, 0b1011])
    self.assertSequenceEqual(result, [0.5j, 0j])

  def test_cirq_qsim_simulate_fullstate(self):
    # Pick qubits.
    a, b, c, d = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0)
    ]

    # Create a circuit.
    cirq_circuit = cirq.Circuit(
        cirq.Moment([
            cirq.X(a)**0.5,  # Square root of X.
            cirq.H(b),       # Hadamard.
            cirq.X(c),       # X.
            cirq.H(d),       # Hadamard.
        ]),
        cirq.Moment([
            cirq.X(a)**0.5,  # Square root of X.
            cirq.CX(b, c),   # ControlX.
            cirq.S(d),       # S (square root of Z).
        ]),
        cirq.Moment([
            cirq.I(a),
            cirq.ISWAP(b, c),
        ])
    )

    qsim_circuit = qsimcirq.QSimCircuit(cirq_circuit)

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(qsim_circuit, qubit_order=[a, b, c, d])
    assert result.state_vector().shape == (16,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=[a, b, c, d])
    # When using rotation gates such as S, qsim may add a global phase relative
    # to other simulators. This is fine, as the result is equivalent.
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())

  def test_cirq_qsim_simulate_random_unitary(self):

    q0, q1 = cirq.LineQubit.range(2)
    qsimSim = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 0})
    for iter in range(10):
        random_circuit = cirq.testing.random_circuit(qubits=[q0, q1],
                                                     n_moments=8,
                                                     op_density=0.99,
                                                     random_state=iter)

        cirq.ConvertToCzAndSingleGates().optimize_circuit(random_circuit) # cannot work with params
        cirq.ExpandComposite().optimize_circuit(random_circuit)
        qsim_circuit = qsimcirq.QSimCircuit(random_circuit)

        result = qsimSim.simulate(qsim_circuit, qubit_order=[q0, q1])
        assert result.state_vector().shape == (4,)

        cirqSim = cirq.Simulator()
        cirq_result = cirqSim.simulate(random_circuit, qubit_order=[q0, q1])
        # When using rotation gates such as S, qsim may add a global phase relative
        # to other simulators. This is fine, as the result is equivalent.
        assert cirq.linalg.allclose_up_to_global_phase(
            result.state_vector(),
            cirq_result.state_vector(),
            atol = 1.e-6
        )

  def test_cirq_qsimh_simulate(self):
    # Pick qubits.
    a, b = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

    # Create a circuit
    cirq_circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.X(a))

    qsim_circuit = qsimcirq.QSimCircuit(cirq_circuit)

    qsimh_options = {'k': [0], 'w': 0, 'p': 1, 'r': 1}
    qsimhSim = qsimcirq.QSimhSimulator(qsimh_options)
    result = qsimhSim.compute_amplitudes(
        qsim_circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
    self.assertSequenceEqual(result, [0j, 0j, (1 + 0j), 0j])


if __name__ == '__main__':
  unittest.main()
