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

from typing import Sequence

import cirq

import qsimcirq.qsim_circuit as qsimc

from . import qsim


class QSimhSimulator(cirq.SimulatesAmplitudes):
    def __init__(self, qsimh_options: dict = None):
        if qsimh_options is None:
            qsimh_options = {}
        """Creates a new QSimhSimulator using the given options.

        Args:
            qsim_options: A map of circuit options for the simulator. These will be
                applied to all circuits run using this simulator. Accepted keys and
                their behavior are as follows:
                    - 'k': Comma-separated list of ints. Indices of "part 1" qubits.
                    - 'p': int (>= 0). Number of "prefix" gates.
                    - 'r': int (>= 0). Number of "root" gates.
                    - 't': int (> 0). Number of threads to run on. Default: 1.
                    - 'v': int (>= 0). Log verbosity. Default: 0.
                    - 'w': int (>= 0). Prefix value.
                See qsim/docs/usage.md for more details on these options.
        """
        self.qsimh_options = {"t": 1, "f": 2, "v": 0}
        self.qsimh_options.update(qsimh_options)

    def compute_amplitudes_sweep(
        self,
        program: cirq.Circuit,
        bitstrings: Sequence[int],
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:

        if not isinstance(program, qsimc.QSimCircuit):
            program = qsimc.QSimCircuit(program)

        n_qubits = len(program.all_qubits())
        # qsim numbers qubits in reverse order from cirq
        bitstrings = [
            format(bitstring, "b").zfill(n_qubits)[::-1] for bitstring in bitstrings
        ]

        options = {"i": "\n".join(bitstrings)}
        options.update(self.qsimh_options)
        param_resolvers = cirq.to_resolvers(params)

        trials_results = []
        for prs in param_resolvers:

            solved_circuit = cirq.resolve_parameters(program, prs)

            options["c"], _ = solved_circuit.translate_cirq_to_qsim(qubit_order)

            options.update(self.qsimh_options)
            amplitudes = qsim.qsimh_simulate(options)
            trials_results.append(amplitudes)

        return trials_results
