import cirq, qsimcirq

# Create a Bell state, |00) + |11)
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))

sim = qsimcirq.QSimSimulator()
result = sim.run(circuit, repetitions=1000)
# Outputs a histogram dict of result:count pairs.
# Expected result is a bunch of 0s and 3s, with no 1s or 2s.
print(result.histogram(key="m"))
