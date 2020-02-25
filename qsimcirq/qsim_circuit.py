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

import cirq


class QSimCircuit(cirq.Circuit):

  def __init__(self,
               cirq_circuit: cirq.Circuit,
               device: cirq.devices = cirq.devices.UNCONSTRAINED_DEVICE,
               allow_decomposition: bool = False):

    if allow_decomposition:
      super().__init__([], device)
      for moment in cirq_circuit:
        for op in moment:
          # This should call decompose on the gates
          self.append(op)
    else:
      super().__init__(cirq_circuit, device)

  def __eq__(self, other):
    if not isinstance(other, QSimCircuit):
      return False
    # equality is tested, for the moment, for cirq.Circuit
    return super().__eq__(other)

  def _resolve_parameters_(self, param_resolver: cirq.study.ParamResolver):

    qsim_circuit = super()._resolve_parameters_(param_resolver)

    qsim_circuit.device = self.device

    return qsim_circuit

  def translate_cirq_to_qsim(
      self,
      qubit_order: cirq.ops.QubitOrderOrList = cirq.ops.QubitOrder.DEFAULT
  ) -> str:
    """
        Translates this Cirq circuit to the qsim representation
        :qubit_order: Ordering of qubits
        :return: the string representing line-by-line the qsim circuit
        """

    circuit_data = []
    ordered_qubits = cirq.ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        self.all_qubits())
    qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
    for mi, moment in enumerate(self):
      for op in moment:

        qub_str = ""
        for qub in op.qubits:
          qub_str += "{} ".format(qubit_to_index_dict[qub])

        qsim_gate = ""
        qsim_params = ""
        if isinstance(op.gate, cirq.ops.HPowGate)\
                and op.gate.exponent == 1.0:
          qsim_gate = "h"
        elif isinstance(op.gate, cirq.ops.ZPowGate) \
                and op.gate.exponent == 0.25:
          qsim_gate = "t"
        elif isinstance(op.gate, cirq.ops.XPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "x"
        elif isinstance(op.gate, cirq.ops.YPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "y"
        elif isinstance(op.gate, cirq.ops.ZPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "z"
        elif isinstance(op.gate, cirq.ops.XPowGate) \
                and op.gate.exponent == 0.5:
          qsim_gate = "x_1_2"
        elif isinstance(op.gate, cirq.ops.YPowGate) \
                and op.gate.exponent == 0.5:
          qsim_gate = "y_1_2"
        elif isinstance(op.gate, cirq.ops.XPowGate):
          qsim_gate = "rx"
          qsim_params = str(op.gate.exponent)
        elif isinstance(op.gate, cirq.ops.YPowGate):
          qsim_gate = "ry"
          qsim_params = str(op.gate.exponent)
        elif isinstance(op.gate, cirq.ops.ZPowGate):
          qsim_gate = "rz"
          qsim_params = str(op.gate.exponent)
        elif isinstance(op.gate, cirq.ops.CZPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "cz"
        elif isinstance(op.gate, cirq.ops.CNotPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "cnot"
        elif isinstance(op.gate, cirq.ops.ISwapPowGate) \
                and op.gate.exponent == 1.0:
          qsim_gate = "is"
        elif isinstance(op.gate, cirq.ops.FSimGate):
          qsim_gate = "fs"
          qsim_params = "{} {}".format(op.gate.theta, op.gate.phi)
        else:
          raise ValueError("{!r} No translation for ".format(op))

        # The moment is missing
        qsim_gate = "{} {} {} {}".format(mi, qsim_gate, qub_str.strip(),
                                         qsim_params.strip())
        circuit_data.append(qsim_gate.strip())

    circuit_data.insert(0, str(len(ordered_qubits)))
    return "\n".join(circuit_data)
