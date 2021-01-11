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

import numpy as np
import sympy
import unittest
import cirq
import qsimcirq


class MainTest(unittest.TestCase):

  def test_cirq_too_big_gate(self):
    # Pick qubits.
    a, b, c, d, e, f, g = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 0),
    ]

    # Create a circuit with a gate larger than 6 qubits.
    cirq_circuit = cirq.Circuit(cirq.IdentityGate(7).on(a, b, c, d, e, f, g))

    qsimSim = qsimcirq.QSimSimulator()
    with self.assertRaises(NotImplementedError):
      qsimSim.compute_amplitudes(cirq_circuit, bitstrings=[0b0, 0b1])


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

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.compute_amplitudes(
        cirq_circuit, bitstrings=[0b0100, 0b1011])
    assert np.allclose(result, [0.5j, 0j])


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

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=[a, b, c, d])
    assert result.state_vector().shape == (16,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=[a, b, c, d])
    # When using rotation gates such as S, qsim may add a global phase relative
    # to other simulators. This is fine, as the result is equivalent.
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_cirq_qsim_simulate_sweep(self):
    # Pick qubits.
    a, b = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
    ]
    x = sympy.Symbol('x')

    # Create a circuit.
    cirq_circuit = cirq.Circuit(
        cirq.Moment([
            cirq.X(a)**x,
            cirq.H(b),       # Hadamard.
        ]),
        cirq.Moment([
            cirq.CX(a, b),   # ControlX.
        ]),
    )

    params = [{x: 0.25}, {x: 0.5}, {x: 0.75}]
    qsimSim = qsimcirq.QSimSimulator()
    qsim_result = qsimSim.simulate_sweep(cirq_circuit, params)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate_sweep(cirq_circuit, params)

    for i in range(len(qsim_result)):
      assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result[i].state_vector(), cirq_result[i].state_vector())

    # initial_state supports bitstrings.
    qsim_result = qsimSim.simulate_sweep(cirq_circuit, params,
                                         initial_state=0b01)
    cirq_result = cirqSim.simulate_sweep(cirq_circuit, params,
                                         initial_state=0b01)
    for i in range(len(qsim_result)):
      assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result[i].state_vector(), cirq_result[i].state_vector())

    # initial_state supports state vectors.
    initial_state = np.asarray([0.5j, 0.5, -0.5j, -0.5], dtype=np.complex64)
    qsim_result = qsimSim.simulate_sweep(
      cirq_circuit, params, initial_state=initial_state)
    cirq_result = cirqSim.simulate_sweep(
      cirq_circuit, params, initial_state=initial_state)
    for i in range(len(qsim_result)):
      assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result[i].state_vector(), cirq_result[i].state_vector())

  def test_input_vector_validation(self):
    cirq_circuit = cirq.Circuit(
      cirq.X(cirq.LineQubit(0)), cirq.X(cirq.LineQubit(1))
    )
    params = [{}]
    qsimSim = qsimcirq.QSimSimulator()

    with self.assertRaises(ValueError):
      initial_state = np.asarray([0.25]*16, dtype=np.complex64)
      qsim_result = qsimSim.simulate_sweep(
        cirq_circuit, params, initial_state=initial_state)

    with self.assertRaises(TypeError):
      initial_state = np.asarray([0.5]*4)
      qsim_result = qsimSim.simulate_sweep(
        cirq_circuit, params, initial_state=initial_state)


  def test_cirq_qsim_run(self):
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
        cirq.CZ(a, d),  # ControlZ.
        # measure qubits
        cirq.measure(a, key='ma'),
        cirq.measure(b, key='mb'),
        cirq.measure(c, key='mc'),
        cirq.measure(d, key='md'),
    )
    qsimSim = qsimcirq.QSimSimulator()
    assert isinstance(qsimSim, cirq.SimulatesSamples)

    result = qsimSim.run(cirq_circuit, repetitions=5)
    for key, value in result.measurements.items():
      assert(value.shape == (5, 1))


  def test_qsim_run_vs_cirq_run(self):
    # Simple circuit, want to check mapping of qubit(s) to their measurements
    a, b, c, d = [
      cirq.GridQubit(0, 0),
      cirq.GridQubit(0, 1),
      cirq.GridQubit(1, 0),
      cirq.GridQubit(1, 1),
    ]
    circuit = cirq.Circuit(
        cirq.X(b),
        cirq.CX(b, d),
        cirq.measure(a, b, c, key='mabc'),
        cirq.measure(d, key='md'),
    )

    # run in cirq
    simulator = cirq.Simulator()
    cirq_result = simulator.run(circuit, repetitions=20)

    # run in qsim
    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.run(circuit, repetitions=20)

    # are they the same?
    assert(qsim_result == cirq_result)


  def test_intermediate_measure(self):
    # Demonstrate that intermediate measurement is possible.
    a, b = [
      cirq.GridQubit(0, 0),
      cirq.GridQubit(0, 1),
    ]
    circuit = cirq.Circuit(
      cirq.X(a), cirq.CX(a, b), cirq.measure(a, b, key='m1'),
      cirq.CZ(a, b), cirq.measure(a, b, key='m2'),
      cirq.X(a), cirq.CX(a, b), cirq.measure(a, b, key='m3'),
      # Trailing gates with no measurement do not affect results.
      cirq.H(a), cirq.H(b),
    )

    simulator = cirq.Simulator()
    cirq_result = simulator.run(circuit, repetitions=20)

    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.run(circuit, repetitions=20)

    assert(qsim_result == cirq_result)


  def test_sampling_nondeterminism(self):
    # Ensure that reusing a QSimSimulator doesn't reuse the original seed.
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))
    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.run(circuit, repetitions=100)

    result_counts = qsim_result.histogram(key='m')
    assert(result_counts[0] > 1)
    assert(result_counts[1] > 1)


  def test_matrix1_gate(self):
    q = cirq.LineQubit(0)
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)

    cirq_circuit = cirq.Circuit(cirq.MatrixGate(m).on(q))
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit)
    assert result.state_vector().shape == (2,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_matrix2_gate(self):
    qubits = cirq.LineQubit.range(2)
    m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    cirq_circuit = cirq.Circuit(cirq.MatrixGate(m).on(*qubits))
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (4,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_big_matrix_gates(self):
    qubits = cirq.LineQubit.range(3)
    # Toffoli gate as a matrix.
    m = np.array([
      [1, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1, 0],
    ])

    cirq_circuit = cirq.Circuit(
      cirq.H(qubits[0]), cirq.H(qubits[1]),
      cirq.MatrixGate(m).on(*qubits),
    )
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (8,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_decompose_to_matrix_gates(self):

    class UnknownThreeQubitGate(cirq.ops.Gate):
      """This gate is not recognized by qsim, and cannot be decomposed.
      
      qsim should attempt to convert it to a MatrixGate to resolve the issue.
      """
      def __init__(self):
        pass

      def _num_qubits_(self):
        return 3

      def _qid_shape_(self):
        return (2, 2, 2)

      def _unitary_(self):
        # Toffoli gate as a matrix.
        return np.array([
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1, 0],
        ])

    qubits = cirq.LineQubit.range(3)
    cirq_circuit = cirq.Circuit(
      cirq.H(qubits[0]), cirq.H(qubits[1]),
      UnknownThreeQubitGate().on(*qubits),
    )
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (8,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_basic_controlled_gate(self):
    qubits = cirq.LineQubit.range(3)

    cirq_circuit = cirq.Circuit(
      cirq.H(qubits[1]), cirq.Y(qubits[2]),
      cirq.X(qubits[0]).controlled_by(qubits[1]),
      cirq.CX(*qubits[1:]).controlled_by(qubits[0]),
      cirq.H(qubits[1]).controlled_by(qubits[0], qubits[2]),
    )
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (8,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_controlled_matrix_gates(self):
    qubits = cirq.LineQubit.range(4)
    m1 = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    m2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    cirq_circuit = cirq.Circuit(
      cirq.MatrixGate(m1).on(qubits[0]).controlled_by(qubits[3]),
      cirq.MatrixGate(m2).on(*qubits[1:3]).controlled_by(qubits[0]),
      cirq.MatrixGate(m1).on(qubits[2]).controlled_by(qubits[0], qubits[1],
                                                      qubits[3]),
      cirq.MatrixGate(m2).on(qubits[0], qubits[3]).controlled_by(*qubits[1:3]),
    )
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (16,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_control_values(self):
    qubits = cirq.LineQubit.range(3)

    cirq_circuit = cirq.Circuit(
      # Controlled by |01) state on qubits 1 and 2
      cirq.X(qubits[0]).controlled_by(*qubits[1:], control_values=[0, 1]),
      # Controlled by either |0) or |1) on qubit 0 (i.e., uncontrolled)
      cirq.X(qubits[1]).controlled_by(qubits[0], control_values=[(0, 1)]),
      # Controlled by |10) state on qubits 0 and 1
      cirq.X(qubits[2]).controlled_by(qubits[1], qubits[0],
                                      control_values=[0, 1]),
    )
    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (8,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())

    qubits = cirq.LineQid.for_qid_shape([2, 3, 2])
    cirq_circuit = cirq.Circuit(
      # Controlled by |12) state on qubits 0 and 1
      # Since qsim does not support qudits (yet), this gate is omitted.
      cirq.X(qubits[2]).controlled_by(*qubits[:2], control_values=[1, 2]),
    )
    qsimSim = qsimcirq.QSimSimulator()
    with self.assertWarnsRegex(RuntimeWarning,
                               'Gate has no valid control value'):
      result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector()[0] == 1


  def test_decomposable_gate(self):
    qubits = cirq.LineQubit.range(4)

    # The Toffoli gate (CCX) decomposes into multiple qsim-supported gates.
    cirq_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        cirq.Moment(
          cirq.CCX(*qubits[:3]),
          cirq.H(qubits[3]),
        ),
        cirq.H(qubits[2]),
        cirq.H(qubits[3]),
    )

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (16,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    # Decomposition may result in gates which add a global phase.
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_complicated_decomposition(self):
    qubits = cirq.LineQubit.range(4)

    # The QFT gate decomposes cleanly into the qsim gateset.
    cirq_circuit = cirq.Circuit(
        cirq.QuantumFourierTransformGate(4).on(*qubits))

    qsimSim = qsimcirq.QSimSimulator()
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert result.state_vector().shape == (16,)
    cirqSim = cirq.Simulator()
    cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=qubits)
    # Decomposition may result in gates which add a global phase.
    assert cirq.linalg.allclose_up_to_global_phase(
        result.state_vector(), cirq_result.state_vector())


  def test_multi_qubit_fusion(self):
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    qubits = [q0, q1, q2, q3]
    cirq_circuit = cirq.Circuit(
      cirq.CX(q0, q1), cirq.X(q2)**0.5, cirq.Y(q3)**0.5,
      cirq.CX(q0, q2), cirq.T(q1), cirq.T(q3),
      cirq.CX(q1, q2), cirq.X(q3)**0.5, cirq.Y(q0)**0.5,
      cirq.CX(q1, q3), cirq.T(q0), cirq.T(q2),
      cirq.CX(q2, q3), cirq.X(q0)**0.5, cirq.Y(q1)**0.5,
    )

    qsimSim = qsimcirq.QSimSimulator(qsim_options={'f': 2})
    result_2q_fusion = qsimSim.simulate(cirq_circuit, qubit_order=qubits)

    qsimSim = qsimcirq.QSimSimulator(qsim_options={'f': 4})
    result_4q_fusion = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
    assert cirq.linalg.allclose_up_to_global_phase(
        result_2q_fusion.state_vector(), result_4q_fusion.state_vector())


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

        result = qsimSim.simulate(random_circuit, qubit_order=[q0, q1])
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

    qsimh_options = {'k': [0], 'w': 0, 'p': 1, 'r': 1}
    qsimhSim = qsimcirq.QSimhSimulator(qsimh_options)
    result = qsimhSim.compute_amplitudes(
        cirq_circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
    assert np.allclose(result, [0j, 0j, (1 + 0j), 0j])


  def test_cirq_qsim_params(self):
    qubit = cirq.GridQubit(0,0)

    circuit = cirq.Circuit(cirq.X(qubit)**sympy.Symbol("beta"))
    params = cirq.ParamResolver({'beta': 0.5})

    simulator = cirq.Simulator()
    cirq_result = simulator.simulate(circuit, param_resolver = params)

    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.simulate(circuit, param_resolver = params)

    assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result.state_vector(), cirq_result.state_vector())


  def test_cirq_qsim_all_supported_gates(self):
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(0, 1)
    q3 = cirq.GridQubit(0, 0)

    circuit = cirq.Circuit(
      cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
      ]),
      cirq.Moment([
        cirq.T(q0),
        cirq.T(q1),
        cirq.T(q2),
        cirq.T(q3),
      ]),
      cirq.Moment([
        cirq.CZPowGate(exponent=0.7, global_shift=0.2)(q0, q1),
        cirq.CXPowGate(exponent=1.2, global_shift=0.4)(q2, q3),
      ]),
      cirq.Moment([
        cirq.XPowGate(exponent=0.3, global_shift=1.1)(q0),
        cirq.YPowGate(exponent=0.4, global_shift=1)(q1),
        cirq.ZPowGate(exponent=0.5, global_shift=0.9)(q2),
        cirq.HPowGate(exponent=0.6, global_shift=0.8)(q3),
      ]),
      cirq.Moment([
        cirq.CX(q0, q2),
        cirq.CZ(q1, q3),
      ]),
      cirq.Moment([
        cirq.X(q0),
        cirq.Y(q1),
        cirq.Z(q2),
        cirq.S(q3),
      ]),
      cirq.Moment([
        cirq.XXPowGate(exponent=0.4, global_shift=0.7)(q0, q1),
        cirq.YYPowGate(exponent=0.8, global_shift=0.5)(q2, q3),
      ]),
      cirq.Moment([
        cirq.I(q0),
        cirq.I(q1),
        cirq.IdentityGate(2)(q2, q3)
      ]),
      cirq.Moment([
        cirq.rx(0.7)(q0),
        cirq.ry(0.2)(q1),
        cirq.rz(0.4)(q2),
        cirq.PhasedXPowGate(
            phase_exponent=0.8, exponent=0.6, global_shift=0.3)(q3),
      ]),
      cirq.Moment([
        cirq.ZZPowGate(exponent=0.3, global_shift=1.3)(q0, q2),
        cirq.ISwapPowGate(exponent=0.6, global_shift=1.2)(q1, q3),
      ]),
      cirq.Moment([
        cirq.XPowGate(exponent=0.1, global_shift=0.9)(q0),
        cirq.YPowGate(exponent=0.2, global_shift=1)(q1),
        cirq.ZPowGate(exponent=0.3, global_shift=1.1)(q2),
        cirq.HPowGate(exponent=0.4, global_shift=1.2)(q3),
      ]),
      cirq.Moment([
        cirq.SwapPowGate(exponent=0.2, global_shift=0.9)(q0, q1),
        cirq.PhasedISwapPowGate(phase_exponent = 0.8, exponent=0.6)(q2, q3),
      ]),
      cirq.Moment([
        cirq.PhasedXZGate(
            x_exponent=0.2, z_exponent=0.3, axis_phase_exponent=1.4)(q0),
        cirq.T(q1),
        cirq.H(q2),
        cirq.S(q3),
      ]),
      cirq.Moment([
        cirq.SWAP(q0, q2),
        cirq.XX(q1, q3),
      ]),
      cirq.Moment([
        cirq.rx(0.8)(q0),
        cirq.ry(0.9)(q1),
        cirq.rz(1.2)(q2),
        cirq.T(q3),
      ]),
      cirq.Moment([
        cirq.YY(q0, q1),
        cirq.ISWAP(q2, q3),
      ]),
      cirq.Moment([
        cirq.T(q0),
        cirq.Z(q1),
        cirq.Y(q2),
        cirq.X(q3),
      ]),
      cirq.Moment([
        cirq.FSimGate(0.3, 1.7)(q0, q2),
        cirq.ZZ(q1, q3),
      ]),
      cirq.Moment([
        cirq.ry(1.3)(q0),
        cirq.rz(0.4)(q1),
        cirq.rx(0.7)(q2),
        cirq.S(q3),
      ]),
      cirq.Moment([
        cirq.IdentityGate(4).on(q0, q1, q2, q3),
      ]),
      cirq.Moment([
        cirq.CCZPowGate(exponent=0.7, global_shift=0.3)(q2, q0, q1),
      ]),
      cirq.Moment([
        cirq.CCXPowGate(exponent=0.4, global_shift=0.6)(
          q3, q1, q0).controlled_by(q2, control_values=[0]),
      ]),
      cirq.Moment([
        cirq.rx(0.3)(q0),
        cirq.ry(0.5)(q1),
        cirq.rz(0.7)(q2),
        cirq.rx(0.9)(q3),
      ]),
      cirq.Moment([
        cirq.TwoQubitDiagonalGate([0.1, 0.2, 0.3, 0.4])(q0, q1),
      ]),
      cirq.Moment([
        cirq.ThreeQubitDiagonalGate([0.5, 0.6, 0.7, 0.8,
                                     0.9, 1, 1.2, 1.3])(q1, q2, q3),
      ]),
      cirq.Moment([
          cirq.CSwapGate()(q0, q3, q1),
      ]),
      cirq.Moment([
        cirq.rz(0.6)(q0),
        cirq.rx(0.7)(q1),
        cirq.ry(0.8)(q2),
        cirq.rz(0.9)(q3),
      ]),
      cirq.Moment([
        cirq.TOFFOLI(q3, q2, q0),
      ]),
      cirq.Moment([
        cirq.FREDKIN(q1, q3, q2),
      ]),
      cirq.Moment([
        cirq.MatrixGate(np.array([[0, -0.5 - 0.5j, -0.5 - 0.5j, 0],
                                  [0.5 - 0.5j, 0, 0, -0.5 + 0.5j],
                                  [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
                                  [0, -0.5 - 0.5j, 0.5 + 0.5j, 0]]))(q0, q1),
        cirq.MatrixGate(np.array([[0.5 - 0.5j, 0, 0, -0.5 + 0.5j],
                                  [0, 0.5 - 0.5j, -0.5 + 0.5j, 0],
                                  [0, -0.5 + 0.5j, -0.5 + 0.5j, 0],
                                  [0.5 - 0.5j, 0, 0, 0.5 - 0.5j]]))(q2, q3),
      ]),
      cirq.Moment([
        cirq.MatrixGate(np.array([[1, 0], [0, 1j]]))(q0),
        cirq.MatrixGate(np.array([[0, -1j], [1j, 0]]))(q1),
        cirq.MatrixGate(np.array([[0, 1], [1, 0]]))(q2),
        cirq.MatrixGate(np.array([[1, 0], [0, -1]]))(q3),
      ]),
      cirq.Moment([
        cirq.riswap(0.7)(q0, q1),
        cirq.givens(1.2)(q2, q3),
      ]),
      cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
      ]),
    )

    simulator = cirq.Simulator()
    cirq_result = simulator.simulate(circuit)

    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.simulate(circuit)

    assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result.state_vector(), cirq_result.state_vector())


  def test_cirq_qsim_global_shift(self):
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(0, 1)
    q3 = cirq.GridQubit(0, 0)

    circuit = cirq.Circuit(
      cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
      ]),
      cirq.Moment([
        cirq.CXPowGate(exponent=1, global_shift=0.7)(q0, q1),
        cirq.CZPowGate(exponent=1, global_shift=0.9)(q2, q3),
      ]),
      cirq.Moment([
        cirq.XPowGate(exponent=1, global_shift=1.1)(q0),
        cirq.YPowGate(exponent=1, global_shift=1)(q1),
        cirq.ZPowGate(exponent=1, global_shift=0.9)(q2),
        cirq.HPowGate(exponent=1, global_shift=0.8)(q3),
      ]),
      cirq.Moment([
        cirq.XXPowGate(exponent=1, global_shift=0.2)(q0, q1),
        cirq.YYPowGate(exponent=1, global_shift=0.3)(q2, q3),
      ]),
      cirq.Moment([
        cirq.ZPowGate(exponent=0.25, global_shift=0.4)(q0),
        cirq.ZPowGate(exponent=0.5, global_shift=0.5)(q1),
        cirq.YPowGate(exponent=1, global_shift=0.2)(q2),
        cirq.ZPowGate(exponent=1, global_shift=0.3)(q3),
      ]),
      cirq.Moment([
        cirq.ZZPowGate(exponent=1, global_shift=0.2)(q0, q1),
        cirq.SwapPowGate(exponent=1, global_shift=0.3)(q2, q3),
      ]),
      cirq.Moment([
        cirq.XPowGate(exponent=1, global_shift=0)(q0),
        cirq.YPowGate(exponent=1, global_shift=0)(q1),
        cirq.ZPowGate(exponent=1, global_shift=0)(q2),
        cirq.HPowGate(exponent=1, global_shift=0)(q3),
      ]),
      cirq.Moment([
        cirq.ISwapPowGate(exponent=1, global_shift=0.3)(q0, q1),
        cirq.ZZPowGate(exponent=1, global_shift=0.5)(q2, q3),
      ]),
      cirq.Moment([
        cirq.ZPowGate(exponent=0.5, global_shift=0)(q0),
        cirq.ZPowGate(exponent=0.25, global_shift=0)(q1),
        cirq.XPowGate(exponent=0.9, global_shift=0)(q2),
        cirq.YPowGate(exponent=0.8, global_shift=0)(q3),
      ]),
      cirq.Moment([
        cirq.CZPowGate(exponent=0.3, global_shift=0)(q0, q1),
        cirq.CXPowGate(exponent=0.4, global_shift=0)(q2, q3),
      ]),
      cirq.Moment([
        cirq.ZPowGate(exponent=1.3, global_shift=0)(q0),
        cirq.HPowGate(exponent=0.8, global_shift=0)(q1),
        cirq.XPowGate(exponent=0.9, global_shift=0)(q2),
        cirq.YPowGate(exponent=0.4, global_shift=0)(q3),
      ]),
      cirq.Moment([
        cirq.XXPowGate(exponent=0.8, global_shift=0)(q0, q1),
        cirq.YYPowGate(exponent=0.6, global_shift=0)(q2, q3),
      ]),
      cirq.Moment([
        cirq.HPowGate(exponent=0.7, global_shift=0)(q0),
        cirq.ZPowGate(exponent=0.2, global_shift=0)(q1),
        cirq.YPowGate(exponent=0.3, global_shift=0)(q2),
        cirq.XPowGate(exponent=0.7, global_shift=0)(q3),
      ]),
      cirq.Moment([
        cirq.ZZPowGate(exponent=0.1, global_shift=0)(q0, q1),
        cirq.SwapPowGate(exponent=0.6, global_shift=0)(q2, q3),
      ]),
      cirq.Moment([
        cirq.XPowGate(exponent=0.4, global_shift=0)(q0),
        cirq.YPowGate(exponent=0.3, global_shift=0)(q1),
        cirq.ZPowGate(exponent=0.2, global_shift=0)(q2),
        cirq.HPowGate(exponent=0.1, global_shift=0)(q3),
      ]),
      cirq.Moment([
        cirq.ISwapPowGate(exponent=1.3, global_shift=0)(q0, q1),
        cirq.CXPowGate(exponent=0.5, global_shift=0)(q2, q3),
      ]),
      cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
      ]),
    )

    simulator = cirq.Simulator()
    cirq_result = simulator.simulate(circuit)

    qsim_simulator = qsimcirq.QSimSimulator()
    qsim_result = qsim_simulator.simulate(circuit)

    assert cirq.linalg.allclose_up_to_global_phase(
        qsim_result.state_vector(), cirq_result.state_vector())


if __name__ == '__main__':
  unittest.main()
