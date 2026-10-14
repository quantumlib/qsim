# Copyright 2026 Google LLC. All Rights Reserved.
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

"""Cirq sampler backed by qsim's matrix product state simulator."""

from typing import Sequence

import cirq
import numpy as np

import qsimcirq.qsim_circuit as qsimc
from qsimcirq import qsim


class QSimMPSSimulator(cirq.Sampler):
    """Samples terminal measurements using qsim's MPS simulator.

    The MPS backend currently supports circuits with at least two qubits
    and one- or two-qubit unitary gates. Measurements must be terminal.
    Repeated measurement keys, controlled operations, channels, and gates
    acting on more than two qubits are not supported.
    """

    def __init__(self, seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None):
        self._prng = cirq.value.parse_random_state(seed)

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
    ) -> Sequence[cirq.Result]:
        if repetitions < 0:
            raise ValueError("repetitions must be non-negative.")

        if not isinstance(program, cirq.AbstractCircuit):
            program = cirq.Circuit(program)

        if not program.are_all_measurements_terminal():
            raise ValueError("MPS sampling requires terminal measurements.")

        all_qubits = cirq.QubitOrder.DEFAULT.order_for(program.all_qubits())
        if len(all_qubits) < 2:
            raise ValueError("MPS sampling requires at least two qubits.")

        measurement_ops = [
            op
            for op in program.all_operations()
            if isinstance(op.gate, cirq.MeasurementGate)
        ]

        measurements_by_key = {}
        for op in measurement_ops:
            key = cirq.measurement_key_name(op)
            if key in measurements_by_key:
                raise ValueError(
                    f"MPS sampling does not support repeated measurement "
                    f"key {key!r} yet."
                )
            measurements_by_key[key] = op

        # Remove measurements from the circuit passed to the native backend.
        # Add identities to preserve qubits that otherwise become idle.
        unitary_circuit = cirq.Circuit(
            op
            for op in program.all_operations()
            if not isinstance(op.gate, cirq.MeasurementGate)
        )
        unitary_circuit.append(cirq.I(qubit) for qubit in all_qubits)

        results = []
        for resolver in cirq.to_resolvers(params):
            resolved_circuit = cirq.resolve_parameters(unitary_circuit, resolver)

            # Validate after resolving parameters, so parameterized unitary
            # gates can be accepted when their parameters are resolved.
            for op in resolved_circuit.all_operations():
                if isinstance(op, cirq.ControlledOperation):
                    raise ValueError(
                        "MPS sampling does not support controlled operations yet."
                    )
                if len(op.qubits) > 2:
                    raise ValueError(
                        "MPS sampling supports one- and two-qubit gates only."
                    )
                if not cirq.has_unitary(op):
                    raise ValueError(
                        f"MPS sampling supports unitary gates only; got {op!r}."
                    )

            translated, _ = qsimc.QSimCircuit(resolved_circuit).translate_cirq_to_qsim()

            raw_results = qsim.qsim_mps_sample(
                {"c": translated, "s": int(self._prng.randint(2**31 - 1))},
                repetitions,
            )

            num_qubits = len(all_qubits)
            full_results = np.array(
                [
                    [bool(result & (1 << q)) for q in reversed(range(num_qubits))]
                    for result in raw_results
                ],
                dtype=bool,
            ).reshape(repetitions, num_qubits)

            measurement_results = {}
            qubit_map = {qubit: i for i, qubit in enumerate(all_qubits)}
            for key, op in measurements_by_key.items():
                indices = [qubit_map[qubit] for qubit in op.qubits]
                invert_mask = np.asarray(op.gate.full_invert_mask(), dtype=bool)
                measurement_results[key] = (
                    full_results[:, indices] ^ invert_mask
                ).astype(np.uint8)

            results.append(
                cirq.ResultDict(params=resolver, measurements=measurement_results)
            )

        return results
