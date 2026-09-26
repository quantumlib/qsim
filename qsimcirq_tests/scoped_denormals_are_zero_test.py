# Copyright 2026 Google LLC
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

import ctypes

import cirq
import pytest

import qsimcirq


@pytest.mark.parametrize("denormals_are_zeros", [False, True, None])
def test_denormals_are_zeros_restored(denormals_are_zeros):
    orig_value = ctypes.c_float(1e-40).value

    if denormals_are_zeros is None:
        simulator = qsimcirq.QSimSimulator()
    else:
        simulator = qsimcirq.QSimSimulator(
            qsim_options=qsimcirq.QSimOptions(
                denormals_are_zeros=denormals_are_zeros,
            ),
        )

    simulator.simulate(cirq.Circuit())
    assert ctypes.c_float(1e-40).value == orig_value


@pytest.mark.parametrize("denormals_are_zeros", [False, True])
def test_denormals_are_zeros_restored_after_expectation_values(
    denormals_are_zeros,
):
    orig_value = ctypes.c_float(1e-40).value
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(qubit))

    result = qsimcirq.QSimSimulator(
        qsim_options=qsimcirq.QSimOptions(
            denormals_are_zeros=denormals_are_zeros,
        ),
    ).simulate_expectation_values(circuit, [cirq.Z(qubit)])

    assert result == [pytest.approx(0.0)]
    assert ctypes.c_float(1e-40).value == orig_value
