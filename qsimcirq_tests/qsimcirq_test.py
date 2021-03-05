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
import cirq
import pytest
import qsimcirq


class NoiseTrigger(cirq.SingleQubitGate):
  """A no-op gate with no _unitary_ method defined.
  
  Appending this gate to a circuit will force it to use qtrajectory, but the
  new circuit will otherwise behave identically to the original.
  """
  # def _mixture_(self):
  #   return ((1.0, np.asarray([1, 0, 0, 1])),)

  def _channel_(self):
    return (np.asarray([1, 0, 0, 1]),)


def test_cirq_too_big_gate():
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
  with pytest.raises(NotImplementedError):
    qsimSim.compute_amplitudes(cirq_circuit, bitstrings=[0b0, 0b1])


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_cirq_qsim_simulate(mode: str):
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

  if mode == 'noisy':
    cirq_circuit.append(NoiseTrigger().on(a))

  qsimSim = qsimcirq.QSimSimulator()
  result = qsimSim.compute_amplitudes(
      cirq_circuit, bitstrings=[0b0100, 0b1011])
  assert np.allclose(result, [0.5j, 0j])


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_cirq_qsim_simulate_fullstate(mode: str):
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

  if mode == 'noisy':
    cirq_circuit.append(NoiseTrigger().on(a))

  qsimSim = qsimcirq.QSimSimulator()
  result = qsimSim.simulate(cirq_circuit, qubit_order=[a, b, c, d])
  assert result.state_vector().shape == (16,)
  cirqSim = cirq.Simulator()
  cirq_result = cirqSim.simulate(cirq_circuit, qubit_order=[a, b, c, d])
  # When using rotation gates such as S, qsim may add a global phase relative
  # to other simulators. This is fine, as the result is equivalent.
  assert cirq.linalg.allclose_up_to_global_phase(
      result.state_vector(), cirq_result.state_vector())


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_cirq_qsim_simulate_sweep(mode: str):
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

  if mode == 'noisy':
    cirq_circuit.append(NoiseTrigger().on(a))

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


def test_input_vector_validation():
  cirq_circuit = cirq.Circuit(
    cirq.X(cirq.LineQubit(0)), cirq.X(cirq.LineQubit(1))
  )
  params = [{}]
  qsimSim = qsimcirq.QSimSimulator()

  with pytest.raises(ValueError):
    initial_state = np.asarray([0.25]*16, dtype=np.complex64)
    qsim_result = qsimSim.simulate_sweep(
      cirq_circuit, params, initial_state=initial_state)

  with pytest.raises(TypeError):
    initial_state = np.asarray([0.5]*4)
    qsim_result = qsimSim.simulate_sweep(
      cirq_circuit, params, initial_state=initial_state)


def test_numpy_params():
  q0 = cirq.LineQubit(0)
  x, y = sympy.Symbol('x'), sympy.Symbol('y')
  circuit = cirq.Circuit(cirq.X(q0) ** x, cirq.H(q0) ** y)
  prs = [{x: np.int64(0), y: np.int64(1)}, {x: np.int64(1), y: np.int64(0)}]

  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_result = qsim_simulator.simulate_sweep(circuit, params=prs)


def test_invalid_params():
  # Parameters must have numeric values.
  q0 = cirq.LineQubit(0)
  x, y = sympy.Symbol('x'), sympy.Symbol('y')
  circuit = cirq.Circuit(cirq.X(q0) ** x, cirq.H(q0) ** y)
  prs = [{x: np.int64(0), y: np.int64(1)}, {x: np.int64(1), y: 'z'}]

  qsim_simulator = qsimcirq.QSimSimulator()
  with pytest.raises(ValueError, match='Parameters must be numeric'):
    _ = qsim_simulator.simulate_sweep(circuit, params=prs)


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_cirq_qsim_run(mode: str):
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
  if mode == 'noisy':
    cirq_circuit.append(NoiseTrigger().on(a))

  qsimSim = qsimcirq.QSimSimulator()
  assert isinstance(qsimSim, cirq.SimulatesSamples)

  result = qsimSim.run(cirq_circuit, repetitions=5)
  for key, value in result.measurements.items():
    assert(value.shape == (5, 1))


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_qsim_run_vs_cirq_run(mode: str):
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

  if mode == 'noisy':
    circuit.append(NoiseTrigger().on(a))

  # run in cirq
  simulator = cirq.Simulator()
  cirq_result = simulator.run(circuit, repetitions=20)

  # run in qsim
  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_result = qsim_simulator.run(circuit, repetitions=20)

  # are they the same?
  assert(qsim_result == cirq_result)


# TODO: test against Cirq simulator when Cirq v0.10 is released.
@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_expectation_values(mode: str):
  a, b = [
    cirq.GridQubit(0, 0),
    cirq.GridQubit(0, 1),
  ]
  x_exp = sympy.Symbol('x_exp')
  h_exp = sympy.Symbol('h_exp')
  circuit = cirq.Circuit(
    cirq.X(a) ** x_exp, cirq.H(b),
    cirq.H(a) ** h_exp, cirq.H(b) ** h_exp,
  )
  params = [
    {x_exp: 0, h_exp: 0},  # |0+)
    {x_exp: 1, h_exp: 0},  # |1+)
    {x_exp: 0, h_exp: 1},  # |+0)
    {x_exp: 1, h_exp: 1},  # |-0)
  ]
  psum1 = cirq.Z(a) + 3 * cirq.X(b)
  psum2 = cirq.X(a) - 3 * cirq.Z(b)

  if mode == 'noisy':
    circuit.append(NoiseTrigger().on(a))

  qsim_simulator = qsimcirq.QSimSimulator()
  with pytest.raises(ValueError, match='terminal measurement checking'):
    _ = qsim_simulator.simulate_expectation_values_sweep(
      circuit, [psum1, psum2], params)
    
  qsim_result = qsim_simulator.simulate_expectation_values_sweep(
    circuit, [psum1, psum2], params, permit_terminal_measurements=True)
  assert len(qsim_result) == 4
  assert len(qsim_result[0]) == 2

  expected_result = [[4, 0], [2, 0], [0, -2], [0, -4]]
  assert cirq.approx_eq(qsim_result, expected_result, atol=1e-6)


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_intermediate_measure(mode: str):
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

  if mode == 'noisy':
    circuit.append(NoiseTrigger().on(a))

  simulator = cirq.Simulator()
  cirq_result = simulator.run(circuit, repetitions=20)

  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_result = qsim_simulator.run(circuit, repetitions=20)

  assert(qsim_result == cirq_result)


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_sampling_nondeterminism(mode: str):
  # Ensure that reusing a QSimSimulator doesn't reuse the original seed.
  q = cirq.GridQubit(0, 0)
  circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))
  if mode == 'noisy':
    circuit.append(NoiseTrigger().on(q))

  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_result = qsim_simulator.run(circuit, repetitions=100)

  result_counts = qsim_result.histogram(key='m')
  assert(result_counts[0] > 1)
  assert(result_counts[1] > 1)


def test_matrix1_gate():
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


def test_matrix2_gate():
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


def test_big_matrix_gates():
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


def test_decompose_to_matrix_gates():

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


def test_basic_controlled_gate():
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


def test_controlled_matrix_gates():
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


def test_control_values():
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
  with pytest.warns(RuntimeWarning, match='Gate has no valid control value'):
    result = qsimSim.simulate(cirq_circuit, qubit_order=qubits)
  assert result.state_vector()[0] == 1


def test_decomposable_gate():
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


def test_complicated_decomposition():
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


# Helper class for noisy circuit tests.
class NoiseStep(cirq.Gate):
  def __init__(self, matrix, num_qubits=1):
    self._matrix = matrix
    self._num_qubits = num_qubits

  def _num_qubits_(self):
    return self._num_qubits

  def _unitary_(self):
    # Not necessarily a unitary.
    return self._matrix

  def __str__(self):
    return f'NoiseStep({self._matrix})'

  def __repr__(self):
    return str(self)


def test_mixture_simulation():
  q0, q1 = cirq.LineQubit.range(2)
  pflip = cirq.phase_flip(p=0.4)
  bflip = cirq.bit_flip(p=0.6)
  cirq_circuit = cirq.Circuit(
    cirq.X(q0) ** 0.5, cirq.X(q1) ** 0.5,
    pflip.on(q0),
    bflip.on(q1),
  )

  possible_circuits = [
    cirq.Circuit(cirq.X(q0) ** 0.5, cirq.X(q1) ** 0.5, pf, bf)
    # Extract the operators from the mixtures to construct trajectories.
    for pf in [NoiseStep(m).on(q0) for m in cirq.channel(pflip)]
    for bf in [NoiseStep(m).on(q1) for m in cirq.channel(bflip)]
  ]
  possible_states = [
    cirq.Simulator().simulate(pc).state_vector()
    for pc in possible_circuits
  ]
  # Since some "gates" were non-unitary, we must normalize.
  possible_states = [ps / np.linalg.norm(ps) for ps in possible_states]

  # Minimize flaky tests with a fixed seed.
  qsimSim = qsimcirq.QSimSimulator(seed=1)
  result_hist = [0] * len(possible_states)
  run_count = 100
  for _ in range(run_count):
    result = qsimSim.simulate(cirq_circuit, qubit_order=[q0, q1])
    for i, ps in enumerate(possible_states):
      if cirq.allclose_up_to_global_phase(result.state_vector(), ps):
        result_hist[i] += 1
        break

  # Each observed result should match one of the possible_results.
  assert sum(result_hist) == run_count
  # Over 100 runs, it's reasonable to expect all four outcomes.
  assert all(result_count > 0 for result_count in result_hist)


def test_channel_simulation():
  q0, q1 = cirq.LineQubit.range(2)
  # These probabilities are set unreasonably high in order to reduce the number
  # of runs required to observe every possible operator.
  amp_damp = cirq.amplitude_damp(gamma=0.5)
  gen_amp_damp = cirq.generalized_amplitude_damp(p=0.4, gamma=0.6)
  cirq_circuit = cirq.Circuit(
    cirq.X(q0) ** 0.5, cirq.X(q1) ** 0.5,
    amp_damp.on(q0), gen_amp_damp.on(q1),
  )

  possible_circuits = [
    cirq.Circuit(cirq.X(q0) ** 0.5, cirq.X(q1) ** 0.5, ad, gad)
    # Extract the operators from the channels to construct trajectories.
    for ad in [NoiseStep(m).on(q0) for m in cirq.channel(amp_damp)]
    for gad in [NoiseStep(m).on(q1) for m in cirq.channel(gen_amp_damp)]
  ]
  possible_states = [
    cirq.Simulator().simulate(pc).state_vector()
    for pc in possible_circuits
  ]
  # Since some "gates" were non-unitary, we must normalize.
  possible_states = [ps / np.linalg.norm(ps) for ps in possible_states]

  # Minimize flaky tests with a fixed seed.
  qsimSim = qsimcirq.QSimSimulator(seed=1)
  result_hist = [0] * len(possible_states)
  run_count = 200
  for _ in range(run_count):
    result = qsimSim.simulate(cirq_circuit, qubit_order=[q0, q1])
    for i, ps in enumerate(possible_states):
      if cirq.allclose_up_to_global_phase(result.state_vector(), ps):
        result_hist[i] += 1
        break

  # Each observed result should match one of the possible_results.
  assert sum(result_hist) == run_count
  # Over 200 runs, it's reasonable to expect all eight outcomes.
  assert all(result_count > 0 for result_count in result_hist)


# Helper class for multi-qubit noisy circuit tests.
class NoiseChannel(cirq.Gate):
  def __init__(self, *prob_mat_pairs, num_qubits=1):
    self._prob_op_pairs = [
      (prob, NoiseStep(m, num_qubits)) for prob, m in prob_mat_pairs
    ]
    self._num_qubits = num_qubits

  def _num_qubits_(self):
    return self._num_qubits

  def _channel_(self):
    return [cirq.unitary(op) for _, op, in self._prob_op_pairs]

  def steps(self):
    return [m for _, m in self._prob_op_pairs]

  def __str__(self):
    return f'NoiseChannel({self._ops})'

  def __repr__(self):
    return str(self)

# Helper class for multi-qubit noisy circuit tests.
class NoiseMixture(NoiseChannel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _mixture_(self):
    return [(prob, cirq.unitary(op)) for prob, op, in self._prob_op_pairs]


@pytest.mark.parametrize('cx_qubits', [
  [cirq.LineQubit(0), cirq.LineQubit(1)],
  [cirq.LineQubit(0), cirq.LineQubit(2)],
  [cirq.LineQubit(1), cirq.LineQubit(0)],
  [cirq.LineQubit(1), cirq.LineQubit(2)],
  [cirq.LineQubit(2), cirq.LineQubit(0)],
  [cirq.LineQubit(2), cirq.LineQubit(1)],
])
@pytest.mark.parametrize('noise_type', [NoiseMixture, NoiseChannel])
def test_multi_qubit_noise(cx_qubits, noise_type):
  # Tests that noise across multiple qubits works correctly.
  qs = cirq.LineQubit.range(3)
  for q in qs:
    if q not in cx_qubits:
      q_no_cx = q
      break

  ambiguous_cx = noise_type(
    # CX(*cx_qubits)
    (
      0.5,  # prob
      np.asarray([
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 0, 0,
      ]) / np.sqrt(2),
    ),
    # CX(*cx_qubits)
    (
      0.5,  # prob
      np.asarray([
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0,
        0, 0, 1, 0,
      ]) / np.sqrt(2),
    ),
    num_qubits=2,
  )
  cirq_circuit = cirq.Circuit(
    cirq.X(cx_qubits[0]) ** 0.5, cirq.X(q_no_cx),
    ambiguous_cx.on(*cx_qubits),
  )

  possible_circuits = [
    cirq.Circuit(cirq.X(cx_qubits[0]) ** 0.5, cirq.X(q_no_cx), cx)
    # Extract the operators from the mixture to construct trajectories.
    for cx in [step.on(*cx_qubits) for step in ambiguous_cx.steps()]
  ]
  possible_states = [
    cirq.Simulator().simulate(pc).state_vector()
    for pc in possible_circuits
  ]
  # Since some "gates" were non-unitary, we must normalize.
  possible_states = [ps / np.linalg.norm(ps) for ps in possible_states]

  # Minimize flaky tests with a fixed seed.
  qsimSim = qsimcirq.QSimSimulator(seed=1)
  result_hist = [0] * len(possible_states)
  run_count = 20
  for _ in range(run_count):
    result = qsimSim.simulate(cirq_circuit, qubit_order=qs)
    for i, ps in enumerate(possible_states):
      if cirq.allclose_up_to_global_phase(result.state_vector(), ps):
        result_hist[i] += 1
        break

  # Each observed result should match one of the possible_results.
  assert sum(result_hist) == run_count
  # Over 20 runs, it's reasonable to expect both outcomes.
  assert all(result_count > 0 for result_count in result_hist)


def test_noise_aggregation():
  q0 = cirq.LineQubit(0)
  # damp_prob is set high to minimize test variance.
  # Even with this setting, estimation of states and expectation values from
  # noisy circuits is highly variable, so this test uses wide tolerances.
  damp_prob = 0.4
  circuit = cirq.Circuit(
    cirq.X(q0), cirq.amplitude_damp(gamma=damp_prob).on(q0),
  )
  psum1 = cirq.Z(q0)
  psum2 = cirq.X(q0)

  # Test expectation value aggregation over repetitions of a noisy circuit.
  # Repetitions are handled in C++, so overhead costs are minimal.
  qsim_simulator = qsimcirq.QSimSimulator(qsim_options={"r": 10000}, seed=1)
  qsim_evs = qsim_simulator.simulate_expectation_values_sweep(
    circuit, [psum1, psum2], params={}, permit_terminal_measurements=True)
  assert len(qsim_evs) == 1
  assert len(qsim_evs[0]) == 2

  # <Z> = (-1) * (probability of |1>) + 1 * (probability of |0>)
  # For damp_prob = 0.4, <Z> == -0.2
  damped_zval = damp_prob - (1 - damp_prob)
  expected_evs = [[damped_zval, 0]]
  assert cirq.approx_eq(qsim_evs, expected_evs, atol=0.05)


def test_multi_qubit_fusion():
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


@pytest.mark.parametrize('mode', ['noiseless', 'noisy'])
def test_cirq_qsim_simulate_random_unitary(mode: str):

  q0, q1 = cirq.LineQubit.range(2)
  qsimSim = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 0})
  for iter in range(10):
      random_circuit = cirq.testing.random_circuit(qubits=[q0, q1],
                                                    n_moments=8,
                                                    op_density=0.99,
                                                    random_state=iter)

      cirq.ConvertToCzAndSingleGates().optimize_circuit(random_circuit) # cannot work with params
      cirq.ExpandComposite().optimize_circuit(random_circuit)
      if mode == 'noisy':
        random_circuit.append(NoiseTrigger().on(q0))

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


def test_cirq_qsimh_simulate():
  # Pick qubits.
  a, b = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

  # Create a circuit
  cirq_circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.X(a))

  qsimh_options = {'k': [0], 'w': 0, 'p': 1, 'r': 1}
  qsimhSim = qsimcirq.QSimhSimulator(qsimh_options)
  result = qsimhSim.compute_amplitudes(
      cirq_circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
  assert np.allclose(result, [0j, 0j, (1 + 0j), 0j])


def test_cirq_qsim_params():
  qubit = cirq.GridQubit(0,0)

  circuit = cirq.Circuit(cirq.X(qubit)**sympy.Symbol("beta"))
  params = cirq.ParamResolver({'beta': 0.5})

  simulator = cirq.Simulator()
  cirq_result = simulator.simulate(circuit, param_resolver = params)

  qsim_simulator = qsimcirq.QSimSimulator()
  qsim_result = qsim_simulator.simulate(circuit, param_resolver = params)

  assert cirq.linalg.allclose_up_to_global_phase(
      qsim_result.state_vector(), cirq_result.state_vector())


def test_cirq_qsim_all_supported_gates():
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


def test_cirq_qsim_global_shift():
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
