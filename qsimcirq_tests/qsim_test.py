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

import unittest
import numpy as np
from qsimcirq import qsim


class MainTest(unittest.TestCase):

  def test_qsim_simulate(self):
    qsim_options = {
        'c': '2\n0 cnot 0 1\n1 cnot 1 0\n2 cz 0 1\n',
        'i': '00\n01\n10\n11',
        't': 1,
        'v': 0
    }
    self.assertSequenceEqual(
        qsim.qsim_simulate(qsim_options), [(1 + 0j), 0j, 0j, 0j])

  def test_qsim_simulate_fullstate(self):
    qsim_fullstate_options = {
        'c': '2\n0 cnot 0 1\n1 cnot 1 0\n2 cz 0 1\n',
        't': 1,
        'v': 0
    }
    self.assertSequenceEqual(
        qsim.qsim_simulate_fullstate(qsim_fullstate_options).tolist(),
        [1., 0., 0., 0., 0., 0., 0., 0.])

  def test_qsimh_simulate(self):
    qsimh_options = {
        'c': '2\n0 cnot 0 1\n1 cnot 1 0\n2 cz 0 1\n',
        'i': '00\n01\n10\n11',
        'k': [0],
        'w': 0,
        'p': 1,
        'r': 1,
        't': 1,
        'v': 0
    }
    self.assertSequenceEqual(
        qsim.qsimh_simulate(qsimh_options), [(1 + 0j), 0j, 0j, 0j])


if __name__ == '__main__':
  unittest.main()
