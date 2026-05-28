#!/usr/bin/env python3

"""Script to check for maximum number of qubits where QSimSimulator segfaults."""

import gc
import sys

import cirq.testing
import numpy as np

import qsimcirq

assert __name__ == "__main__"
if len(sys.argv) != 2:
    print(f"usage: {__file__} num_qubits")
    sys.exit(0)

nqubits = int(sys.argv[1])

q = cirq.LineQubit.range(nqubits)
m1 = cirq.Moment(cirq.H.on_each(q))
m2 = cirq.Moment(cirq.CX(qi, qj) for qi, qj in zip(q[0::2], q[1::2]))
m3 = cirq.Moment(cirq.measure(*q))
c = cirq.Circuit(10 * [m1, m2, m3])

my_sim = qsimcirq.QSimSimulator()

gc.disable()
gc.collect()

state_vector = my_sim.simulate(program=c).state_vector()
print(state_vector.shape, state_vector)
