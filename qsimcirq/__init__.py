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


qsim = _load_simd_qsim()
qsim_gpu = _load_qsim_gpu()

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
