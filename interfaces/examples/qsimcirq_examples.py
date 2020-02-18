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

# The interface between Cirq and the Python interface to the C++ qsim
import sys, os
sys.path.insert(
    1, os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../'))
import qsimcirq.qsim_simulator as qsimSimulator
import qsimcirq.qsimh_simulator as qsimhSimulator
import qsimcirq.qsim_circuit as qcirc
from qsimcirq import qsim
import cirq


def qsim_example():
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
  print('Cirq Circuit:')
  print(cirq_circuit)

  qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

  qsimSim = qsimSimulator.QSimSimulator()
  result = qsimSim.compute_amplitudes(
      qsim_circuit, bitstrings=['0100', '1011'])
  print('Output using qsim simulator:')
  print(result)


def qsim_fullstate_example():
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
  print('Cirq Circuit:')
  print(cirq_circuit)

  qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

  qsimSim = qsimSimulator.QSimSimulator()
  result = qsimSim.simulate(qsim_circuit)
  print('Output using qsim fullstate simulator:')
  print(result)


def qsimh_example():
  # Pick qubits.
  a, b = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

  # Create a circuit
  cirq_circuit = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CZ(a, b))
  print('Cirq Circuit:')
  print(cirq_circuit)

  qsim_circuit = qcirc.QSimCircuit(cirq_circuit)

  qsimh_options = {
      'k': [0],
      'w': 0,
      'p': 1,
      'r': 1
  }
  qsimhSim = qsimhSimulator.QSimhSimulator(qsimh_options)
  result = qsimhSim.compute_amplitudes(
      qsim_circuit, bitstrings=['00', '01', '10', '11'])
  print('Output using qsimh simulator:')
  print(result)


if __name__ == '__main__':
  print('======================')
  qsim_example()
  print('======================')
  qsim_fullstate_example()
  print('======================')
  qsimh_example()
