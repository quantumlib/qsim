import cirq
import qsimcirq
import time

circuit = cirq.testing.random_circuit(
    qubits=3, n_moments=3, op_density=1, random_state=11
)

# Display the noiseless circuit.
print("Circuit without noise:")
print(circuit)

# Add noise to the circuit.
noisy = circuit.with_noise(cirq.depolarize(p=0.01))

# Display it.
print("\nCircuit with noise:")
print(noisy)