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

import cirq
import numpy as np
import pytest

import qsimcirq


def test_mps_sampler_samples_terminal_measurements():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.I(q1),
        cirq.I(q2),
        cirq.measure(q0, q1, q2, key="m"),
    )

    result = qsimcirq.QSimMPSSimulator(seed=123).run(circuit, repetitions=5)

    assert result.measurements["m"].shape == (5, 3)
    np.testing.assert_array_equal(result.measurements["m"], np.tile([1, 0, 0], (5, 1)))


def test_mps_sampler_applies_invert_mask():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.I(q0),
        cirq.I(q1),
        cirq.I(q2),
        cirq.measure(q0, q1, key="m", invert_mask=[True, False]),
    )

    result = qsimcirq.QSimMPSSimulator(seed=123).run(circuit, repetitions=4)

    np.testing.assert_array_equal(result.measurements["m"], np.tile([1, 0], (4, 1)))


def test_mps_sampler_rejects_nonterminal_measurements():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Moment(cirq.measure(q0, key="m")),
        cirq.Moment(cirq.X(q0)),
        cirq.Moment(cirq.I(q1), cirq.I(q2)),
    )

    with pytest.raises(ValueError, match="terminal measurements"):
        qsimcirq.QSimMPSSimulator().run(circuit, repetitions=1)


def test_mps_sampler_rejects_controlled_operations():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.ControlledGate(cirq.X).on(q0, q1),
        cirq.I(q2),
        cirq.measure(q0, key="m"),
    )

    with pytest.raises(ValueError, match="controlled operations"):
        qsimcirq.QSimMPSSimulator().run(circuit, repetitions=1)


def test_mps_sampler_rejects_repeated_measurement_keys():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key="m"),
        cirq.measure(q1, key="m"),
        cirq.I(q2),
    )

    with pytest.raises(ValueError, match="repeated measurement key"):
        qsimcirq.QSimMPSSimulator().run(circuit, repetitions=1)


def test_mps_sampler_requires_at_least_two_qubits():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="m"))

    with pytest.raises(ValueError, match="at least two qubits"):
        qsimcirq.QSimMPSSimulator().run(circuit, repetitions=1)


def test_mps_sampler_supports_zero_repetitions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.I(q0),
        cirq.I(q1),
        cirq.measure(q0, key="m"),
    )

    result = qsimcirq.QSimMPSSimulator(seed=123).run(circuit, repetitions=0)

    assert result.measurements["m"].shape == (0, 1)
