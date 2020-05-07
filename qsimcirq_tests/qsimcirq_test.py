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

from typing import List, Union, Sequence, Dict, Optional, TYPE_CHECKING

import unittest
import cirq
import qsimcirq
import random

DEFAULT_GATE_DOMAIN: Dict[cirq.ops.Gate, int] = {
    cirq.ops.CNOT: 2,
    cirq.ops.CZ: 2,
    cirq.ops.H: 1,
    cirq.ops.ISWAP: 2,
    cirq.ops.CZPowGate(): 2,
    cirq.ops.S: 1,
    cirq.ops.SWAP: 2,
    cirq.ops.T: 1,
    cirq.ops.X: 1,
    cirq.ops.Y: 1,
    cirq.ops.Z: 1
}


def random_circuit(qubits: Union[Sequence[cirq.ops.Qid], int],
                   n_moments: int,
                   op_density: float,
                   gate_domain: Optional[Dict[cirq.ops.Gate, int]] = None,
                   random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
                  ) -> cirq.Circuit:
    """Generates a random circuit.

    Args:
        qubits: If a sequence of qubits, then these are the qubits that
            the circuit should act on. Because the qubits on which an
            operation acts are chosen randomly, not all given qubits
            may be acted upon. If an int, then this number of qubits will
            be automatically generated.
        n_moments: the number of moments in the generated circuit.
        op_density: the expected proportion of qubits that are acted on in any
            moment.
        gate_domain: The set of gates to choose from, with a specified arity.
        random_state: Random state or random state seed.

    Raises:
        ValueError:
            * op_density is not in (0, 1).
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density < 1:
        raise ValueError('op_density must be in (0, 1).')
    if gate_domain is None:
        gate_domain = DEFAULT_GATE_DOMAIN
    if not gate_domain:
        raise ValueError('gate_domain must be non-empty')
    max_arity = max(gate_domain.values())

    if isinstance(qubits, int):
        qubits = tuple(cirq.ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError('At least one qubit must be specified.')

    prng = cirq.value.parse_random_state(random_state)

    moments: List[cirq.ops.Moment] = []
    gate_arity_pairs = sorted(gate_domain.items(), key=repr)
    num_gates = len(gate_domain)
    for _ in range(n_moments):
        operations = []
        free_qubits = set(qubits)
        while len(free_qubits) >= max_arity:
            gate, arity = gate_arity_pairs[prng.randint(num_gates)]
            op_qubits = prng.choice(sorted(free_qubits),
                                    size=arity,
                                    replace=False)
            free_qubits.difference_update(op_qubits)
            if prng.rand() <= op_density:
                operations.append(gate(*op_qubits) ** random.uniform(0, 1))
        moments.append(cirq.ops.Moment(operations))

    return cirq.Circuit(moments)


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

    num_qubits = 20
    q = cirq.LineQubit.range(num_qubits)
    qsimSim = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 0})
    for iter in range(10):
        rc = random_circuit(qubits=q, n_moments=8, op_density=0.99, random_state=iter)

        cirq.ConvertToCzAndSingleGates().optimize_circuit(rc) # cannot work with params
        cirq.ExpandComposite().optimize_circuit(rc)
        qsim_circuit = qsimcirq.QSimCircuit(rc)

        result = qsimSim.simulate(qsim_circuit, qubit_order=q)
        assert result.state_vector().shape == (2 ** num_qubits,)

        cirqSim = cirq.Simulator()
        cirq_result = cirqSim.simulate(rc, qubit_order=q)
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
