import unittest
import numpy as np
import cirq
import cirq_qsim.qsim_simulator as qsimSimulator
import cirq_qsim.qsimh_simulator as qsimhSimulator
import cirq_qsim.qsim_circuit as qcirc


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

    qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

    qsimSim = qsimSimulator.QSimSimulator()
    result = qsimSim.compute_amplitudes(
        qsim_circuit, bitstrings=['0100', '1011'])
    self.assertSequenceEqual(result, [0.5j, 0j])

  def test_cirq_qsim_simulate_fullstate(self):
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

    qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

    qsimSim = qsimSimulator.QSimSimulator()
    result = qsimSim.simulate(qsim_circuit)

  def test_cirq_qsimh_simulate(self):
    # Pick qubits.
    a, b = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

    # Create a circuit
    cirq_circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CZ(a, b))

    qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

    qsimh_options = {'k': [0], 'w': 0, 'p': 1, 'r': 1}
    qsimhSim = qsimhSimulator.QSimhSimulator(qsimh_options)
    result = qsimhSim.compute_amplitudes(
        qsim_circuit, bitstrings=['00', '01', '10', '11'])
    self.assertSequenceEqual(result, [(1 + 0j), 0j, 0j, 0j])


if __name__ == '__main__':
  unittest.main()
