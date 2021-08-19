import cirq
import qsimcirq
import time

q0, q1 = cirq.LineQubit.range(2)

circuit = cirq.Circuit(
    # Perform a Hadamard on both qubits
    cirq.H(q0), cirq.H(q1),
    # Apply amplitude damping to q0 with probability 0.1
    cirq.amplitude_damp(gamma=0.1).on(q0),
    # Apply phase damping to q1 with probability 0.1
    cirq.phase_damp(gamma=0.1).on(q1),
)
qsim_simulator = qsimcirq.QSimSimulator()
results = qsim_simulator.simulate(circuit)
print(results.final_state_vector)


# Simulate measuring at the end of the circuit.
measured_circuit = circuit + cirq.measure(q0, q1, key='m')
measure_results = qsim_simulator.run(measured_circuit, repetitions=5)
print(measure_results)

# Calculate only the amplitudes of the |00) and |01) states.
amp_results = qsim_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])
print(amp_results)

# Calculate only the amplitudes of the |00) and |01) states.
amp_results = qsim_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])
print(amp_results)


# Set the "noisy repetitions" to 100.
# This parameter only affects expectation value calculations.
options = {'r': 100}
# Also set the random seed to get reproducible results.
seed = int(time.time())
print(seed)
ev_simulator = qsimcirq.QSimSimulator(qsim_options=options, seed=seed)
# Define observables to measure: <Z> for q0 and <X> for q1.
pauli_sum1 = cirq.Z(q0)
pauli_sum2 = cirq.X(q1)
# Calculate expectation values for the given observables.
ev_results = ev_simulator.simulate_expectation_values(
    circuit,
    observables=[pauli_sum1, pauli_sum2],
)
print(ev_results)