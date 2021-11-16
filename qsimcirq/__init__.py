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


import importlib
from qsimcirq import qsim_decide


def _load_simd_qsim():
    instr = qsim_decide.detect_instructions()
    if instr == 0:
        qsim = importlib.import_module("qsimcirq.qsim_avx512")
    elif instr == 1:
        qsim = importlib.import_module("qsimcirq.qsim_avx2")
    elif instr == 2:
        qsim = importlib.import_module("qsimcirq.qsim_sse")
    else:
        qsim = importlib.import_module("qsimcirq.qsim_basic")
    return qsim


def _load_qsim_gpu():
    instr = qsim_decide.detect_gpu()
    if instr == 0:
        qsim_gpu = importlib.import_module("qsimcirq.qsim_cuda")
    else:
        qsim_gpu = None
    return qsim_gpu


def _load_qsim_custatevec():
    instr = qsim_decide.detect_custatevec()
    if instr == 1:
        qsim_custatevec = importlib.import_module("qsimcirq.qsim_custatevec")
    else:
        qsim_custatevec = None
    return qsim_custatevec


qsim = _load_simd_qsim()
qsim_gpu = _load_qsim_gpu()
qsim_custatevec = _load_qsim_custatevec()

from .qsim_circuit import add_op_to_opstring, add_op_to_circuit, QSimCircuit
from .qsim_simulator import (
    QSimOptions,
    QSimSimulatorState,
    QSimSimulatorTrialResult,
    QSimSimulator,
)
from .qsimh_simulator import QSimhSimulator

from qsimcirq._version import (
    __version__,
)
