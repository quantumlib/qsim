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


def test_denormals_are_zeros_restored():
    orig_value = ctypes.c_float(1e-40).value

    _ = qsimcirq.QSimSimulator(
        qsim_options=qsimcirq.QSimOptions(denormals_are_zeros=False)
    ).simulate(cirq.Circuit())
    assert (
        ctypes.c_float(1e-40).value == orig_value
    ), "denormals_are_zeros=False failed to restore flags"

    _ = qsimcirq.QSimSimulator(
        qsim_options=qsimcirq.QSimOptions(denormals_are_zeros=True)
    ).simulate(cirq.Circuit())
    assert (
        ctypes.c_float(1e-40).value == orig_value
    ), "denormals_are_zeros=True failed to restore flags"

    _ = qsimcirq.QSimSimulator().simulate(cirq.Circuit())
    assert (
        ctypes.c_float(1e-40).value == orig_value
    ), "default simulation failed to restore flags"
