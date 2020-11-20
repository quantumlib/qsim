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
import warnings

import cirq
from qsimcirq import qsim


def _cirq_gate_kind(gate: cirq.ops.Gate):
  if isinstance(gate, cirq.ops.ControlledGate):
    return _cirq_gate_kind(gate.sub_gate)
  if isinstance(gate, cirq.ops.identity.IdentityGate):
    if gate.num_qubits() == 1:
      return qsim.kI1
    if gate.num_qubits() == 2:
      return qsim.kI2
    if gate.num_qubits() <= 6:
      return qsim.kI
    raise NotImplementedError(
      f'Received identity on {gate.num_qubits()} qubits; '
      + 'only up to 6-qubit gates are supported.')
  if isinstance(gate, cirq.ops.XPowGate):
    # cirq.rx also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kX
    return qsim.kXPowGate
  if isinstance(gate, cirq.ops.YPowGate):
    # cirq.ry also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kY
    return qsim.kYPowGate
  if isinstance(gate, cirq.ops.ZPowGate):
    # cirq.rz also uses this path.
    if gate.global_shift == 0:
      if gate.exponent == 1:
        return qsim.kZ
      if gate.exponent == 0.5:
        return qsim.kS
      if gate.exponent == 0.25:
        return qsim.kT
    return qsim.kZPowGate
  if isinstance(gate, cirq.ops.HPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kH
    return qsim.kHPowGate
  if isinstance(gate, cirq.ops.CZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kCZ
    return qsim.kCZPowGate
  if isinstance(gate, cirq.ops.CXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kCX
    return qsim.kCXPowGate
  if isinstance(gate, cirq.ops.PhasedXPowGate):
    return qsim.kPhasedXPowGate
  if isinstance(gate, cirq.ops.PhasedXZGate):
    return qsim.kPhasedXZGate
  if isinstance(gate, cirq.ops.XXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kXX
    return qsim.kXXPowGate
  if isinstance(gate, cirq.ops.YYPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kYY
    return qsim.kYYPowGate
  if isinstance(gate, cirq.ops.ZZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kZZ
    return qsim.kZZPowGate
  if isinstance(gate, cirq.ops.SwapPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kSWAP
    return qsim.kSwapPowGate
  if isinstance(gate, cirq.ops.ISwapPowGate):
    # cirq.riswap also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kISWAP
    return qsim.kISwapPowGate
  if isinstance(gate, cirq.ops.PhasedISwapPowGate):
    # cirq.givens also uses this path.
    return qsim.kPhasedISwapPowGate
  if isinstance(gate, cirq.ops.FSimGate):
    return qsim.kFSimGate
  if isinstance(gate, cirq.ops.TwoQubitDiagonalGate):
    return qsim.kTwoQubitDiagonalGate
  if isinstance(gate, cirq.ops.ThreeQubitDiagonalGate):
    return qsim.kThreeQubitDiagonalGate
  if isinstance(gate, cirq.ops.CCZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kCCZ
    return qsim.kCCZPowGate
  if isinstance(gate, cirq.ops.CCXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
      return qsim.kCCX
    return qsim.kCCXPowGate
  if isinstance(gate, cirq.ops.CSwapGate):
    return qsim.kCSwapGate
  if isinstance(gate, cirq.ops.MatrixGate):
    if gate.num_qubits() <= 6:
      return qsim.kMatrixGate
    raise NotImplementedError(
      f'Received matrix on {gate.num_qubits()} qubits; '
      + 'only up to 6-qubit gates are supported.')
  if isinstance(gate, cirq.ops.MeasurementGate):
    # needed to inherit SimulatesSamples in sims
    return qsim.kMeasurement
  # Unrecognized gates will be decomposed.
  return None


def _control_details(gate: cirq.ops.ControlledGate, qubits):
  control_qubits = []
  control_values = []
  # TODO: support qudit control
  for i, cvs in enumerate(gate.control_values):
    if 0 in cvs and 1 in cvs:
      # This qubit does not affect control.
      continue
    elif 0 not in cvs and 1 not in cvs:
      # This gate will never trigger.
      warnings.warn(f'Gate has no valid control value: {gate}', RuntimeWarning)
      return (None, None)
    # Either 0 or 1 is in cvs, but not both.
    control_qubits.append(qubits[i])
    if 0 in cvs:
      control_values.append(0)
    elif 1 in cvs:
      control_values.append(1)

  return (control_qubits, control_values)


class QSimCircuit(cirq.Circuit):

  def __init__(self,
               cirq_circuit: cirq.Circuit,
               device: cirq.devices = cirq.devices.UNCONSTRAINED_DEVICE,
               allow_decomposition: bool = False):

    if allow_decomposition:
      super().__init__([], device=device)
      for moment in cirq_circuit:
        for op in moment:
          # This should call decompose on the gates
          self.append(op)
    else:
      super().__init__(cirq_circuit, device=device)

  def __eq__(self, other):
    if not isinstance(other, QSimCircuit):
      return False
    # equality is tested, for the moment, for cirq.Circuit
    return super().__eq__(other)

  def _resolve_parameters_(self, param_resolver: cirq.study.ParamResolver):
    return QSimCircuit(
      super()._resolve_parameters_(param_resolver), device=self.device)

  def translate_cirq_to_qsim(
      self,
      qubit_order: cirq.ops.QubitOrderOrList = cirq.ops.QubitOrder.DEFAULT
  ) -> qsim.Circuit:
    """
        Translates this Cirq circuit to the qsim representation.
        :qubit_order: Ordering of qubits
        :return: a C++ qsim Circuit object
        """

    qsim_circuit = qsim.Circuit()
    qsim_circuit.num_qubits = len(self.all_qubits())
    ordered_qubits = cirq.ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        self.all_qubits())

    # qsim numbers qubits in reverse order from cirq
    ordered_qubits = list(reversed(ordered_qubits))

    def has_qsim_kind(op: cirq.ops.GateOperation):
      return _cirq_gate_kind(op.gate) != None

    def to_matrix(op: cirq.ops.GateOperation):
      mat = cirq.protocols.unitary(op.gate, None)
      if mat is None:
          return NotImplemented
      
      return cirq.ops.MatrixGate(mat).on(*op.qubits)

    qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
    time_offset = 0
    for moment in self:
      ops_by_gate = [
        cirq.decompose(op, fallback_decomposer=to_matrix, keep=has_qsim_kind)
        for op in moment
      ]
      moment_length = max(len(gate_ops) for gate_ops in ops_by_gate)

      # Gates must be added in time order.
      for gi in range(moment_length):
        for gate_ops in ops_by_gate:
          if gi >= len(gate_ops):
            continue
          qsim_op = gate_ops[gi]
          gate_kind = _cirq_gate_kind(qsim_op.gate)
          time = time_offset + gi
          qubits = [qubit_to_index_dict[q] for q in qsim_op.qubits]

          qsim_gate = qsim_op.gate
          qsim_qubits = qubits
          is_controlled = isinstance(qsim_op.gate, cirq.ops.ControlledGate)
          if is_controlled:
            control_qubits, control_values = _control_details(qsim_op.gate,
                                                              qubits)
            if control_qubits is None:
              # This gate has no valid control, and will be omitted.
              continue

            if qsim_gate.num_qubits() > 4:
              raise NotImplementedError(
              f'Received control gate on {gate.num_qubits()} target qubits; '
              + 'only up to 4-qubit gates are supported.')

            qsim_gate = qsim_gate.sub_gate
            qsim_qubits = qubits[qsim_op.gate.num_controls():]

          if gate_kind == qsim.kTwoQubitDiagonalGate or gate_kind == qsim.kThreeQubitDiagonalGate:
            qsim.add_diagonal_gate(time, qsim_qubits,
                                   qsim_gate._diag_angles_radians, qsim_circuit)
          elif gate_kind == qsim.kMatrixGate:
            flatten = lambda l : [val for i in l for val in [i.real, i.imag]]
            qsim.add_matrix_gate(time, qsim_qubits,
                                 flatten(list(cirq.unitary(qsim_gate).flat)),
                                 qsim_circuit)
          else:
            params = {
              p.strip('_'): val for p, val in vars(qsim_gate).items()
              if isinstance(val, float) or isinstance(val, int)
            }
            qsim.add_gate(gate_kind, time, qsim_qubits, params, qsim_circuit)

          if is_controlled:
            qsim.control_last_gate(control_qubits, control_values, qsim_circuit)
      time_offset += moment_length

    return qsim_circuit
