import pytest
import cirq
import ctypes
import qsimcirq

def test_denormals_are_zeros_restored():
    orig_value = ctypes.c_float(1e-40).value

    _ = qsimcirq.QSimSimulator(qsim_options=qsimcirq.QSimOptions(denormals_are_zeros=False)).simulate(cirq.Circuit())
    assert ctypes.c_float(1e-40).value == orig_value, "denormals_are_zeros=False failed to restore flags"

    _ = qsimcirq.QSimSimulator(qsim_options=qsimcirq.QSimOptions(denormals_are_zeros=True)).simulate(cirq.Circuit())
    assert ctypes.c_float(1e-40).value == orig_value, "denormals_are_zeros=True failed to restore flags"

    _ = qsimcirq.QSimSimulator().simulate(cirq.Circuit())
    assert ctypes.c_float(1e-40).value == orig_value, "default simulation failed to restore flags"
