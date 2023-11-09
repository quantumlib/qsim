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

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cirq
import numpy as np

from . import qsim


# List of parameter names that appear in valid Cirq protos.
GATE_PARAMS = [
    "exponent",
    "phase_exponent",
    "global_shift",
    "x_exponent",
    "z_exponent",
    "axis_phase_exponent",
    "phi",
    "theta",
]


def _translate_ControlledGate(gate: cirq.ControlledGate):
    return _cirq_gate_kind(gate.sub_gate)


def _translate_XPowGate(gate: cirq.XPowGate):
    # cirq.rx also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kX
    return qsim.kXPowGate


def _translate_YPowGate(gate: cirq.YPowGate):
    # cirq.ry also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kY
    return qsim.kYPowGate


def _translate_ZPowGate(gate: cirq.ZPowGate):
    # cirq.rz also uses this path.
    if gate.global_shift == 0:
        if gate.exponent == 1:
            return qsim.kZ
        if gate.exponent == 0.5:
            return qsim.kS
        if gate.exponent == 0.25:
            return qsim.kT
    return qsim.kZPowGate


def _translate_HPowGate(gate: cirq.HPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kH
    return qsim.kHPowGate


def _translate_CZPowGate(gate: cirq.CZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kCZ
    return qsim.kCZPowGate


def _translate_CXPowGate(gate: cirq.CXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kCX
    return qsim.kCXPowGate


def _translate_PhasedXPowGate(gate: cirq.PhasedXPowGate):
    return qsim.kPhasedXPowGate


def _translate_PhasedXZGate(gate: cirq.PhasedXZGate):
    return qsim.kPhasedXZGate


def _translate_XXPowGate(gate: cirq.XXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kXX
    return qsim.kXXPowGate


def _translate_YYPowGate(gate: cirq.YYPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kYY
    return qsim.kYYPowGate


def _translate_ZZPowGate(gate: cirq.ZZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kZZ
    return qsim.kZZPowGate


def _translate_SwapPowGate(gate: cirq.SwapPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kSWAP
    return qsim.kSwapPowGate


def _translate_ISwapPowGate(gate: cirq.ISwapPowGate):
    # cirq.riswap also uses this path.
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kISWAP
    return qsim.kISwapPowGate


def _translate_PhasedISwapPowGate(gate: cirq.PhasedISwapPowGate):
    # cirq.givens also uses this path.
    return qsim.kPhasedISwapPowGate


def _translate_FSimGate(gate: cirq.FSimGate):
    return qsim.kFSimGate


def _translate_TwoQubitDiagonalGate(gate: cirq.TwoQubitDiagonalGate):
    return qsim.kTwoQubitDiagonalGate


def _translate_ThreeQubitDiagonalGate(gate: cirq.ThreeQubitDiagonalGate):
    return qsim.kThreeQubitDiagonalGate


def _translate_CCZPowGate(gate: cirq.CCZPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kCCZ
    return qsim.kCCZPowGate


def _translate_CCXPowGate(gate: cirq.CCXPowGate):
    if gate.exponent == 1 and gate.global_shift == 0:
        return qsim.kCCX
    return qsim.kCCXPowGate


def _translate_CSwapGate(gate: cirq.CSwapGate):
    return qsim.kCSwapGate


def _translate_MatrixGate(gate: cirq.MatrixGate):
    if gate.num_qubits() <= 6:
        return qsim.kMatrixGate
    raise NotImplementedError(
        f"Received matrix on {gate.num_qubits()} qubits; "
        + "only up to 6-qubit gates are supported."
    )


def _translate_MeasurementGate(gate: cirq.MeasurementGate):
    # needed to inherit SimulatesSamples in sims
    return qsim.kMeasurement


TYPE_TRANSLATOR = {
    cirq.ControlledGate: _translate_ControlledGate,
    cirq.XPowGate: _translate_XPowGate,
    cirq.YPowGate: _translate_YPowGate,
    cirq.ZPowGate: _translate_ZPowGate,
    cirq.HPowGate: _translate_HPowGate,
    cirq.CZPowGate: _translate_CZPowGate,
    cirq.CXPowGate: _translate_CXPowGate,
    cirq.PhasedXPowGate: _translate_PhasedXPowGate,
    cirq.PhasedXZGate: _translate_PhasedXZGate,
    cirq.XXPowGate: _translate_XXPowGate,
    cirq.YYPowGate: _translate_YYPowGate,
    cirq.ZZPowGate: _translate_ZZPowGate,
    cirq.SwapPowGate: _translate_SwapPowGate,
    cirq.ISwapPowGate: _translate_ISwapPowGate,
    cirq.PhasedISwapPowGate: _translate_PhasedISwapPowGate,
    cirq.FSimGate: _translate_FSimGate,
    cirq.TwoQubitDiagonalGate: _translate_TwoQubitDiagonalGate,
    cirq.ThreeQubitDiagonalGate: _translate_ThreeQubitDiagonalGate,
    cirq.CCZPowGate: _translate_CCZPowGate,
    cirq.CCXPowGate: _translate_CCXPowGate,
    cirq.CSwapGate: _translate_CSwapGate,
    cirq.MatrixGate: _translate_MatrixGate,
    cirq.MeasurementGate: _translate_MeasurementGate,
}


def _cirq_gate_kind(gate: cirq.Gate):
    for gate_type in type(gate).mro():
        translator = TYPE_TRANSLATOR.get(gate_type, None)
        if translator is not None:
            return translator(gate)
    # Unrecognized gates will be decomposed.
    return None


def _has_cirq_gate_kind(op: cirq.Operation):
    if isinstance(op, cirq.ControlledOperation):
        return _has_cirq_gate_kind(op.sub_operation)
    return any(t in TYPE_TRANSLATOR for t in type(op.gate).mro())


def _control_details(
    gate: cirq.ControlledGate, qubits: Sequence[cirq.Qid]
) -> Tuple[List[cirq.Qid], List[int]]:
    control_qubits: List[cirq.Qid] = []
    control_values: List[int] = []
    # TODO: support qudit control
    assignments = list(gate.control_values.expand())
    if len(qubits) > 1 and len(assignments) > 1:
        raise ValueError(
            f"Cannot translate controlled gate with multiple assignments for multiple qubits: {gate}"
        )
    for q, cvs in zip(qubits, zip(*assignments)):
        if 0 in cvs and 1 in cvs:
            # This qubit does not affect control.
            continue
        elif any(cv not in (0, 1) for cv in cvs):
            raise ValueError(
                f"Cannot translate control values other than 0 and 1: cvs={cvs}"
            )
        # Either 0 or 1 is in cvs, but not both.
        control_qubits.append(q)
        if 0 in cvs:
            control_values.append(0)
        elif 1 in cvs:
            control_values.append(1)

    return control_qubits, control_values


def add_op_to_opstring(
    qsim_op: cirq.GateOperation,
    qubit_to_index_dict: Dict[cirq.Qid, int],
    opstring: qsim.OpString,
):
    """Adds an operation to an opstring (observable).

    Raises:
      ValueError if qsim_op is not a single-qubit Pauli (I, X, Y, or Z).
    """
    qsim_gate = qsim_op.gate
    gate_kind = _cirq_gate_kind(qsim_gate)
    if gate_kind not in {qsim.kX, qsim.kY, qsim.kZ, qsim.kI1}:
        raise ValueError(f"OpString should only have Paulis; got {gate_kind}")
    if len(qsim_op.qubits) != 1:
        raise ValueError(f"OpString ops should have 1 qubit; got {len(qsim_op.qubits)}")

    is_controlled = isinstance(qsim_gate, cirq.ControlledGate)
    if is_controlled:
        raise ValueError(f"OpString ops should not be controlled.")

    qubits = [qubit_to_index_dict[q] for q in qsim_op.qubits]
    qsim.add_gate_to_opstring(gate_kind, qubits, opstring)


def add_op_to_circuit(
    qsim_op: cirq.GateOperation,
    time: int,
    qubit_to_index_dict: Dict[cirq.Qid, int],
    circuit: Union[qsim.Circuit, qsim.NoisyCircuit],
):
    """Adds an operation to a noisy or noiseless circuit."""
    qsim_gate = qsim_op.gate
    gate_kind = _cirq_gate_kind(qsim_gate)
    qubits = [qubit_to_index_dict[q] for q in qsim_op.qubits]

    qsim_qubits = qubits
    is_controlled = isinstance(qsim_gate, cirq.ControlledGate)
    if is_controlled:
        control_qubits, control_values = _control_details(
            qsim_gate, qubits[: qsim_gate.num_controls()]
        )
        if control_qubits is None:
            # This gate has no valid control, and will be omitted.
            return

        num_targets = qsim_gate.num_qubits() - qsim_gate.num_controls()
        if num_targets > 4:
            raise NotImplementedError(
                f"Received control gate on {num_targets} target qubits; "
                + "only up to 4-qubit gates are supported."
            )

        qsim_qubits = qubits[qsim_gate.num_controls() :]
        qsim_gate = qsim_gate.sub_gate

    if (
        gate_kind == qsim.kTwoQubitDiagonalGate
        or gate_kind == qsim.kThreeQubitDiagonalGate
    ):
        if isinstance(circuit, qsim.Circuit):
            qsim.add_diagonal_gate(
                time, qsim_qubits, qsim_gate._diag_angles_radians, circuit
            )
        else:
            qsim.add_diagonal_gate_channel(
                time, qsim_qubits, qsim_gate._diag_angles_radians, circuit
            )
    elif gate_kind == qsim.kMatrixGate:
        m = [
            val for i in list(cirq.unitary(qsim_gate).flat) for val in [i.real, i.imag]
        ]
        if isinstance(circuit, qsim.Circuit):
            qsim.add_matrix_gate(time, qsim_qubits, m, circuit)
        else:
            qsim.add_matrix_gate_channel(time, qsim_qubits, m, circuit)
    else:
        params = {}
        for p, val in vars(qsim_gate).items():
            key = p.strip("_")
            if key not in GATE_PARAMS:
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                params[key] = val
            else:
                raise ValueError("Parameters must be numeric.")
        if isinstance(circuit, qsim.Circuit):
            qsim.add_gate(gate_kind, time, qsim_qubits, params, circuit)
        else:
            qsim.add_gate_channel(gate_kind, time, qsim_qubits, params, circuit)

    if is_controlled:
        if isinstance(circuit, qsim.Circuit):
            qsim.control_last_gate(control_qubits, control_values, circuit)
        else:
            qsim.control_last_gate_channel(control_qubits, control_values, circuit)


class QSimCircuit(cirq.Circuit):
    def __init__(
        self,
        cirq_circuit: cirq.Circuit,
        allow_decomposition: bool = False,
    ):
        if allow_decomposition:
            super().__init__()
            for moment in cirq_circuit:
                for op in moment:
                    # This should call decompose on the gates
                    self.append(op)
        else:
            super().__init__(cirq_circuit)
        self._check_for_confusion_matrix()

    def __eq__(self, other):
        if not isinstance(other, QSimCircuit):
            return False
        # equality is tested, for the moment, for cirq.Circuit
        return super().__eq__(other)

    def _resolve_parameters_(
        self, param_resolver: cirq.study.ParamResolver, recursive: bool = True
    ):
        return QSimCircuit(cirq.resolve_parameters(super(), param_resolver, recursive))

    def _check_for_confusion_matrix(self):
        """Checks cirq Circuit for Measurement Gates with confusion matrices.
        Returns:
            Throws a runtime exception if a MeasurementGate with a confusion matrix is included in the circuit
        """
        confusion_maps_on_measurement_gates = [
            op.gate.confusion_map
            for _, op, _ in self.findall_operations_with_gate_type(cirq.MeasurementGate)
            if op.gate.confusion_map
        ]
        for confusion_map in confusion_maps_on_measurement_gates:
            for map_values in confusion_map.values():
                if map_values:
                    raise ValueError(
                        "Confusion Matrices are not currently supported in Qsim. "
                        "See https://github.com/quantumlib/Cirq/issues/6305 for latest status"
                    )

    def translate_cirq_to_qsim(
        self, qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT
    ) -> qsim.Circuit:
        """
        Translates this Cirq circuit to the qsim representation.
        :qubit_order: Ordering of qubits
        :return: a tuple of (C++ qsim Circuit object, moment boundary
            gate indices)
        """

        qsim_circuit = qsim.Circuit()
        ordered_qubits = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits()
        )
        qsim_circuit.num_qubits = len(ordered_qubits)

        # qsim numbers qubits in reverse order from cirq
        ordered_qubits = list(reversed(ordered_qubits))

        def to_matrix(op: cirq.GateOperation):
            mat = cirq.unitary(op.gate, None)
            if mat is None:
                return NotImplemented

            return cirq.MatrixGate(mat).on(*op.qubits)

        qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
        time_offset = 0
        gate_count = 0
        moment_indices = []
        for moment in self:
            ops_by_gate = [
                cirq.decompose(
                    op, fallback_decomposer=to_matrix, keep=_has_cirq_gate_kind
                )
                for op in moment
            ]
            moment_length = max((len(gate_ops) for gate_ops in ops_by_gate), default=0)

            # Gates must be added in time order.
            for gi in range(moment_length):
                for gate_ops in ops_by_gate:
                    if gi >= len(gate_ops):
                        continue
                    qsim_op = gate_ops[gi]
                    time = time_offset + gi
                    add_op_to_circuit(qsim_op, time, qubit_to_index_dict, qsim_circuit)
                    gate_count += 1
            time_offset += moment_length
            moment_indices.append(gate_count)

        return qsim_circuit, moment_indices

    def translate_cirq_to_qtrajectory(
        self, qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT
    ) -> qsim.NoisyCircuit:
        """
        Translates this noisy Cirq circuit to the qsim representation.
        :qubit_order: Ordering of qubits
        :return: a tuple of (C++ qsim NoisyCircuit object, moment boundary
            gate indices)
        """
        qsim_ncircuit = qsim.NoisyCircuit()
        ordered_qubits = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits()
        )

        # qsim numbers qubits in reverse order from cirq
        ordered_qubits = list(reversed(ordered_qubits))

        qsim_ncircuit.num_qubits = len(ordered_qubits)

        def to_matrix(op: cirq.GateOperation):
            mat = cirq.unitary(op.gate, None)
            if mat is None:
                return NotImplemented

            return cirq.MatrixGate(mat).on(*op.qubits)

        qubit_to_index_dict = {q: i for i, q in enumerate(ordered_qubits)}
        time_offset = 0
        gate_count = 0
        moment_indices = []
        for moment in self:
            moment_length = 0
            ops_by_gate = []
            ops_by_mix = []
            ops_by_channel = []
            # Capture ops of each type in the appropriate list.
            for qsim_op in moment:
                if cirq.has_unitary(qsim_op) or cirq.is_measurement(qsim_op):
                    oplist = cirq.decompose(
                        qsim_op, fallback_decomposer=to_matrix, keep=_has_cirq_gate_kind
                    )
                    ops_by_gate.append(oplist)
                    moment_length = max(moment_length, len(oplist))
                    pass
                elif cirq.has_mixture(qsim_op):
                    ops_by_mix.append(qsim_op)
                    moment_length = max(moment_length, 1)
                    pass
                elif cirq.has_kraus(qsim_op):
                    ops_by_channel.append(qsim_op)
                    moment_length = max(moment_length, 1)
                    pass
                else:
                    raise ValueError(f"Encountered unparseable op: {qsim_op}")

            # Gates must be added in time order.
            for gi in range(moment_length):
                # Handle gate output.
                for gate_ops in ops_by_gate:
                    if gi >= len(gate_ops):
                        continue
                    qsim_op = gate_ops[gi]
                    time = time_offset + gi
                    add_op_to_circuit(qsim_op, time, qubit_to_index_dict, qsim_ncircuit)
                    gate_count += 1
                # Only gates decompose to multiple time steps.
                if gi > 0:
                    continue
                # Handle mixture output.
                for mixture in ops_by_mix:
                    mixdata = []
                    for prob, mat in cirq.mixture(mixture):
                        square_mat = np.reshape(mat, (int(np.sqrt(mat.size)), -1))
                        unitary = cirq.is_unitary(square_mat)
                        # Package matrix into a qsim-friendly format.
                        mat = np.reshape(mat, (-1,)).astype(np.complex64, copy=False)
                        mixdata.append((prob, mat.view(np.float32), unitary))
                    qubits = [qubit_to_index_dict[q] for q in mixture.qubits]
                    qsim.add_channel(time_offset, qubits, mixdata, qsim_ncircuit)
                    gate_count += 1
                # Handle channel output.
                for channel in ops_by_channel:
                    chdata = []
                    for i, mat in enumerate(cirq.kraus(channel)):
                        square_mat = np.reshape(mat, (int(np.sqrt(mat.size)), -1))
                        unitary = cirq.is_unitary(square_mat)
                        singular_vals = np.linalg.svd(square_mat)[1]
                        lower_bound_prob = min(singular_vals) ** 2
                        # Package matrix into a qsim-friendly format.
                        mat = np.reshape(mat, (-1,)).astype(np.complex64, copy=False)
                        chdata.append((lower_bound_prob, mat.view(np.float32), unitary))
                    qubits = [qubit_to_index_dict[q] for q in channel.qubits]
                    qsim.add_channel(time_offset, qubits, chdata, qsim_ncircuit)
                    gate_count += 1
            time_offset += moment_length
            moment_indices.append(gate_count)

        return qsim_ncircuit, moment_indices
