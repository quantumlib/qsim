# Copyright 2025 Google LLC. All Rights Reserved.
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
import pytest
from qsimcirq.qsim_simulator import _unpack_results

def test_unpack_int64():
    raw = np.array([0, 1, 2, 3], dtype=np.int64)
    # 0 -> 00
    # 1 -> 01
    # 2 -> 10
    # 3 -> 11
    res = _unpack_results(raw, 2)
    expected = np.array([
        [False, False],
        [False, True],
        [True, False],
        [True, True]
    ], dtype=bool)
    np.testing.assert_array_equal(res, expected)

def test_unpack_uint64():
    # Large uint64 value (MSB set)
    val = (1 << 63) + 1
    raw = np.array([val], dtype=np.uint64)
    res = _unpack_results(raw, 64)

    assert res.shape == (1, 64)
    assert res[0, 0] == True # MSB
    assert res[0, 63] == True # LSB
    assert not res[0, 1] # Check another bit

def test_unpack_object():
    # Arbitrary precision integer > 64 bits
    val = (1 << 100) | 1
    raw = np.array([val], dtype=object)
    res = _unpack_results(raw, 101)

    assert res.shape == (1, 101)
    assert res[0, 0] == True # MSB
    assert res[0, 100] == True # LSB
    assert not res[0, 1]

def test_unpack_list_conversion():
    raw = [0, 3]
    res = _unpack_results(raw, 2)
    expected = np.array([
        [False, False],
        [True, True]
    ], dtype=bool)
    np.testing.assert_array_equal(res, expected)

def test_unpack_empty():
    raw = np.array([], dtype=np.int64)
    res = _unpack_results(raw, 5)
    assert res.shape == (0, 5)
    assert res.dtype == bool

def test_unpack_large_num_qubits_force_object():
    # Even if values are small, if num_qubits > 63, masks must use object dtype
    # to avoid overflow when creating masks.
    num_qubits = 70
    raw = np.array([1], dtype=np.int64)
    res = _unpack_results(raw, num_qubits)

    assert res.shape == (1, num_qubits)
    assert res[0, 69] == True # LSB is set (1)
    assert not res[0, 0] # MSB is not set
