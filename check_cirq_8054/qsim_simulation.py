import gc

import numpy as np

import cirq.testing
import qsimcirq

nqubits = 20
# nqubits = 24

q = cirq.LineQubit.range(nqubits)
m1 = cirq.Moment(cirq.H.on_each(q))
m2 = cirq.Moment(cirq.CX(qi, qj) for qi, qj in zip(q[0::2], q[1::2]))
m3 = cirq.Moment(cirq.measure(*q))
c = cirq.Circuit(10 * [m1, m2, m3])

# make reproducible initial_state
rs = cirq.value.parse_random_state(8054)
initial_state = cirq.testing.random_superposition(2**nqubits, random_state=rs).astype(
    np.complex64
)
my_sim = qsimcirq.QSimSimulator()

gc.disable()
gc.collect()

# burn CPU cycles for a few seconds
for _ in range(50_000_000):
    pass

state_vector = my_sim.simulate(program=c, initial_state=initial_state).state_vector()
print(state_vector.shape, state_vector)

# burn CPU cycles for a few seconds
for _ in range(50_000_000):
    pass
