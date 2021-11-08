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

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from cirq import (
    circuits,
    linalg,
    ops,
    protocols,
    sim,
    study,
    value,
    SimulatesAmplitudes,
    SimulatesFinalState,
    SimulatesSamples,
)

# TODO: import from cirq directly when fix is released
from cirq.sim.simulator import SimulatesExpectationValues

import numpy as np

from . import qsim, qsim_gpu, qsim_custatevec
import qsimcirq.qsim_circuit as qsimc


class QSimSimulatorState(sim.StateVectorSimulatorState):
    def __init__(self, qsim_data: np.ndarray, qubit_map: Dict[ops.Qid, int]):
        state_vector = qsim_data.view(np.complex64)
        super().__init__(state_vector=state_vector, qubit_map=qubit_map)


@value.value_equality(unhashable=True)
class QSimSimulatorTrialResult(sim.StateVectorMixin, sim.SimulationTrialResult):
    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: QSimSimulatorState,
    ):
        super().__init__(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state,
        )

    # The following methods are (temporarily) copied here from
    # cirq.StateVectorTrialResult due to incompatibility with the
    # intermediate state simulation support which that class requires.
    # TODO: remove these methods once inheritance is restored.

    @property
    def final_state_vector(self):
        return self._final_simulator_state.state_vector

    def state_vector(self):
        """Return the state vector at the end of the computation."""
        return self._final_simulator_state.state_vector.copy()

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    def __str__(self) -> str:
        samples = super().__str__()
        final = self.state_vector()
        if len([1 for e in final if abs(e) > 0.001]) < 16:
            state_vector = self.dirac_notation(3)
        else:
            state_vector = str(final)
        return f"measurements: {samples}\noutput vector: {state_vector}"

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text("StateVectorTrialResult(...)")
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        return (
            f"cirq.StateVectorTrialResult(params={self.params!r}, "
            f"measurements={self.measurements!r}, "
            f"final_simulator_state={self._final_simulator_state!r})"
        )


# This should probably live in Cirq...
# TODO: update to support CircuitOperations.
def _needs_trajectories(circuit: circuits.Circuit) -> bool:
    """Checks if the circuit requires trajectory simulation."""
    for op in circuit.all_operations():
        test_op = (
            op
            if not protocols.is_parameterized(op)
            else protocols.resolve_parameters(
                op, {param: 1 for param in protocols.parameter_names(op)}
            )
        )
        if not (protocols.has_unitary(test_op) or protocols.is_measurement(test_op)):
            return True
    return False


@dataclass
class QSimOptions:
    """Container for options to the QSimSimulator.

    Options for the simulator can also be provided as a {string: value} dict,
    using the format shown in the 'as_dict' function for this class.

    Args:
        max_fused_gate_size: maximum number of qubits allowed per fused gate.
            Circuits of less than 22 qubits usually perform best with this set
            to 2 or 3, while larger circuits (with >= 22 qubits) typically
            perform better with it set to 3 or 4.
        cpu_threads: number of threads to use when running on CPU. For best
            performance, this should equal the number of cores on the device.
        ev_noisy_repetitions: number of repetitions used for estimating
            expectation values of a noisy circuit. Does not affect other
            simulation modes.
        use_gpu: whether to use GPU instead of CPU for simulation. The "gpu_*"
            arguments below are only considered if this is set to True.
        gpu_mode: use CUDA if set to 0 (default value) or use the NVIDIA
            cuStateVec library if set to any other value. The "gpu_*"
            arguments below are only considered if this is set to 0.
        gpu_sim_threads: number of threads per CUDA block to use for the GPU
            Simulator. This must be a power of 2 in the range [32, 256].
        gpu_state_threads: number of threads per CUDA block to use for the GPU
            StateSpace. This must be a power of 2 in the range [32, 1024].
        gpu_data_blocks: number of data blocks to use on GPU. Below 16 data
            blocks, performance is noticeably reduced.
        verbosity: Logging verbosity.
        denormals_are_zeros: if true, set flush-to-zero and denormals-are-zeros
            MXCSR control flags. This prevents rare cases of performance
            slowdown potentially at the cost of a tiny precision loss.
    """

    max_fused_gate_size: int = 2
    cpu_threads: int = 1
    ev_noisy_repetitions: int = 1
    use_gpu: bool = False
    gpu_mode: int = 0
    gpu_sim_threads: int = 256
    gpu_state_threads: int = 512
    gpu_data_blocks: int = 16
    verbosity: int = 0
    denormals_are_zeros: bool = False

    def as_dict(self):
        """Generates an options dict from this object.

        Options to QSimSimulator can also be provided in this format directly.
        """
        return {
            "f": self.max_fused_gate_size,
            "t": self.cpu_threads,
            "r": self.ev_noisy_repetitions,
            "g": self.use_gpu,
            "gmode": self.gpu_mode,
            "gsmt": self.gpu_sim_threads,
            "gsst": self.gpu_state_threads,
            "gdb": self.gpu_data_blocks,
            "v": self.verbosity,
            "z": self.denormals_are_zeros,
        }


class QSimSimulator(
    SimulatesSamples,
    SimulatesAmplitudes,
    SimulatesFinalState,
    SimulatesExpectationValues,
):
    def __init__(
        self,
        qsim_options: Union[None, Dict, QSimOptions] = None,
        seed: value.RANDOM_STATE_OR_SEED_LIKE = None,
        circuit_memoization_size: int = 0,
    ):
        """Creates a new QSimSimulator using the given options and seed.

        Args:
            qsim_options: An options dict or QSimOptions object with options
                to use for all circuits run using this simulator. See the
                QSimOptions class for details.
            seed: A random state or seed object, as defined in cirq.value.
            circuit_memoization_size: The number of last translated circuits
                to be memoized from simulation executions, to eliminate
                translation overhead. Every simulation will perform a linear
                search through the list of memoized circuits using circuit
                equality checks, so a large circuit_memoization_size with large
                circuits will incur a significant runtime overhead.
                Note that every resolved parameterization results in a separate
                circuit to be memoized.

        Raises:
            ValueError if internal keys 'c', 'i' or 's' are included in 'qsim_options'.
        """
        if isinstance(qsim_options, QSimOptions):
            qsim_options = qsim_options.as_dict()
        else:
            qsim_options = qsim_options or {}

        if any(k in qsim_options for k in ("c", "i", "s")):
            raise ValueError(
                'Keys {"c", "i", "s"} are reserved for internal use and cannot be '
                "used in QSimCircuit instantiation."
            )
        self._prng = value.parse_random_state(seed)
        self.qsim_options = QSimOptions().as_dict()
        self.qsim_options.update(qsim_options)

        # module to use for simulation
        if self.qsim_options["g"]:
            if self.qsim_options["gmode"] == 0:
                if qsim_gpu is None:
                    raise ValueError(
                        "GPU execution requested, but not supported. If your "
                        "device has GPU support, you may need to compile qsim "
                        "locally."
                    )
                else:
                    self._sim_module = qsim_gpu
            else:
                if qsim_custatevec is None:
                    raise ValueError(
                        "cuStateVec GPU execution requested, but not "
                        "supported. If your device has GPU support and the "
                        "NVIDIA cuStateVec library is installed, you may need "
                        "to compile qsim locally."
                    )
                else:
                    self._sim_module = qsim_custatevec
        else:
            self._sim_module = qsim

        # Deque of (<original cirq circuit>, <translated qsim circuit>) tuples.
        self._translated_circuits = deque(maxlen=circuit_memoization_size)

    def get_seed(self):
        # Limit seed size to 32-bit integer for C++ conversion.
        return self._prng.randint(2 ** 31 - 1)

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            program: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.

        Returns:
            A dictionary from measurement gate key to measurement
            results.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        solved_circuit = protocols.resolve_parameters(circuit, param_resolver)

        return self._sample_measure_results(solved_circuit, repetitions)

    def _sample_measure_results(
        self,
        program: circuits.Circuit,
        repetitions: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Samples from measurement gates in the circuit.

        Note that this will execute the circuit 'repetitions' times.

        Args:
            program: The circuit to sample from.
            repetitions: The number of samples to take.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)

        Raises:
            ValueError: If there are multiple MeasurementGates with the same key,
                or if repetitions is negative.
        """
        if not isinstance(program, qsimc.QSimCircuit):
            program = qsimc.QSimCircuit(program, device=program.device)

        # Compute indices of measured qubits
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(program.all_qubits())
        num_qubits = len(ordered_qubits)

        qubit_map = {qubit: index for index, qubit in enumerate(ordered_qubits)}

        # Computes
        # - the list of qubits to be measured
        # - the start (inclusive) and end (exclusive) indices of each measurement
        # - a mapping from measurement key to measurement gate
        measurement_ops = [
            op
            for _, op, _ in program.findall_operations_with_gate_type(
                ops.MeasurementGate
            )
        ]
        measured_qubits = []  # type: List[ops.Qid]
        bounds = {}  # type: Dict[str, Tuple]
        meas_ops = {}  # type: Dict[str, cirq.GateOperation]
        current_index = 0
        for op in measurement_ops:
            gate = op.gate
            key = protocols.measurement_key_name(gate)
            meas_ops[key] = op
            if key in bounds:
                raise ValueError(f"Duplicate MeasurementGate with key {key}")
            bounds[key] = (current_index, current_index + len(op.qubits))
            measured_qubits.extend(op.qubits)
            current_index += len(op.qubits)

        # Set qsim options
        options = {}
        options.update(self.qsim_options)

        results = {}
        for key, bound in bounds.items():
            results[key] = np.ndarray(
                shape=(repetitions, bound[1] - bound[0]), dtype=int
            )

        noisy = _needs_trajectories(program)
        if noisy:
            translator_fn_name = "translate_cirq_to_qtrajectory"
            sampler_fn = self._sim_module.qtrajectory_sample
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            sampler_fn = self._sim_module.qsim_sample

        if not noisy and program.are_all_measurements_terminal() and repetitions > 1:
            # Measurements must be replaced with identity gates to sample properly.
            # Simply removing them may omit qubits from the circuit.
            for i in range(len(program.moments)):
                program.moments[i] = ops.Moment(
                    op
                    if not isinstance(op.gate, ops.MeasurementGate)
                    else [ops.IdentityGate(1).on(q) for q in op.qubits]
                    for op in program.moments[i]
                )
            translator_fn_name = "translate_cirq_to_qsim"
            options["c"] = self._translate_circuit(
                program,
                translator_fn_name,
                ops.QubitOrder.DEFAULT,
            )
            options["s"] = self.get_seed()
            final_state = self._sim_module.qsim_simulate_fullstate(options, 0)
            full_results = sim.sample_state_vector(
                final_state.view(np.complex64),
                range(num_qubits),
                repetitions=repetitions,
                seed=self._prng,
            )

            for i in range(repetitions):
                for key, op in meas_ops.items():
                    meas_indices = [qubit_map[qubit] for qubit in op.qubits]
                    for j, q in enumerate(meas_indices):
                        results[key][i][j] = full_results[i][q]
        else:
            options["c"] = self._translate_circuit(
                program,
                translator_fn_name,
                ops.QubitOrder.DEFAULT,
            )
            for i in range(repetitions):
                options["s"] = self.get_seed()
                measurements = sampler_fn(options)
                for key, bound in bounds.items():
                    for j in range(bound[1] - bound[0]):
                        results[key][i][j] = int(measurements[bound[0] + j])

        return results

    def compute_amplitudes_sweep(
        self,
        program: circuits.Circuit,
        bitstrings: Sequence[int],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:
        """Computes the desired amplitudes using qsim.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input as an
              string array where each string is formed from measured qubit values
              according to `qubit_order` from most to least significant qubit,
              i.e. in big-endian ordering.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This is
              often used in specifying the initial state, i.e. the ordering of the
              computational basis states.

        Returns:
            List of amplitudes.
        """
        if not isinstance(program, qsimc.QSimCircuit):
            program = qsimc.QSimCircuit(program, device=program.device)

        # qsim numbers qubits in reverse order from cirq
        cirq_order = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            program.all_qubits()
        )
        num_qubits = len(cirq_order)
        bitstrings = [
            format(bitstring, "b").zfill(num_qubits)[::-1] for bitstring in bitstrings
        ]

        options = {"i": "\n".join(bitstrings)}
        options.update(self.qsim_options)

        param_resolvers = study.to_resolvers(params)

        trials_results = []
        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            simulator_fn = self._sim_module.qtrajectory_simulate
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            simulator_fn = self._sim_module.qsim_simulate

        for prs in param_resolvers:
            solved_circuit = protocols.resolve_parameters(program, prs)
            options["c"] = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()
            amplitudes = simulator_fn(options)
            trials_results.append(amplitudes)

        return trials_results

    def simulate_sweep(
        self,
        program: circuits.Circuit,
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Optional[Union[int, np.ndarray]] = None,
    ) -> List["SimulationTrialResult"]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Avoid using this method with `use_gpu=True` in the simulator options;
        when used with GPU this method must copy state from device to host memory
        multiple times, which can be very slow. This issue is not present in
        `simulate_expectation_values_sweep`.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This is
              often used in specifying the initial state, i.e. the ordering of the
              computational basis states.
            initial_state: The initial state for the simulation. This can either
              be an integer representing a pure state (e.g. 11010) or a numpy
              array containing the full state vector. If none is provided, this
              is assumed to be the all-zeros state.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.

        Raises:
            TypeError: if an invalid initial_state is provided.
        """
        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an int or state vector.")
        if not isinstance(program, qsimc.QSimCircuit):
            program = qsimc.QSimCircuit(program, device=program.device)

        options = {}
        options.update(self.qsim_options)

        param_resolvers = study.to_resolvers(params)
        # qsim numbers qubits in reverse order from cirq
        cirq_order = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            program.all_qubits()
        )
        qsim_order = list(reversed(cirq_order))
        num_qubits = len(qsim_order)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64.")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2 ** num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )

        trials_results = []
        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            fullstate_simulator_fn = self._sim_module.qtrajectory_simulate_fullstate
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            fullstate_simulator_fn = self._sim_module.qsim_simulate_fullstate

        for prs in param_resolvers:
            solved_circuit = protocols.resolve_parameters(program, prs)

            options["c"] = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()
            qubit_map = {qubit: index for index, qubit in enumerate(qsim_order)}

            if isinstance(initial_state, int):
                qsim_state = fullstate_simulator_fn(options, initial_state)
            elif isinstance(initial_state, np.ndarray):
                qsim_state = fullstate_simulator_fn(options, input_vector)
            assert qsim_state.dtype == np.float32
            assert qsim_state.ndim == 1
            final_state = QSimSimulatorState(qsim_state, qubit_map)
            # create result for this parameter
            # TODO: We need to support measurements.
            result = QSimSimulatorTrialResult(
                params=prs, measurements={}, final_simulator_state=final_state
            )
            trials_results.append(result)

        return trials_results

    def simulate_expectation_values_sweep(
        self,
        program: "cirq.Circuit",
        observables: Union["cirq.PauliSumLike", List["cirq.PauliSumLike"]],
        params: "study.Sweepable",
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> List[List[float]]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends with
                measurement(s), this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            A list of expectation values, with the value at index `n`
            corresponding to `observables[n]` from the input.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False. (Note: We cannot test this
            until Cirq's `are_any_measurements_terminal` is released.)
        """
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError(
                "Provided circuit has terminal measurements, which may "
                "skew expectation values. If this is intentional, set "
                "permit_terminal_measurements=True."
            )
        if not isinstance(observables, List):
            observables = [observables]
        psumlist = [ops.PauliSum.wrap(pslike) for pslike in observables]

        cirq_order = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            program.all_qubits()
        )
        qsim_order = list(reversed(cirq_order))
        num_qubits = len(qsim_order)
        qubit_map = {qubit: index for index, qubit in enumerate(qsim_order)}

        opsums_and_qubit_counts = []
        for psum in psumlist:
            opsum = []
            opsum_qubits = set()
            for pstr in psum:
                opstring = qsim.OpString()
                opstring.weight = pstr.coefficient
                for q, pauli in pstr.items():
                    op = pauli.on(q)
                    opsum_qubits.add(q)
                    qsimc.add_op_to_opstring(op, qubit_map, opstring)
                opsum.append(opstring)
            opsums_and_qubit_counts.append((opsum, len(opsum_qubits)))

        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an int or state vector.")
        if not isinstance(program, qsimc.QSimCircuit):
            program = qsimc.QSimCircuit(program, device=program.device)

        options = {}
        options.update(self.qsim_options)

        param_resolvers = study.to_resolvers(params)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64.")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2 ** num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )

        results = []
        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            ev_simulator_fn = self._sim_module.qtrajectory_simulate_expectation_values
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            ev_simulator_fn = self._sim_module.qsim_simulate_expectation_values

        for prs in param_resolvers:
            solved_circuit = protocols.resolve_parameters(program, prs)
            options["c"] = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()

            if isinstance(initial_state, int):
                evs = ev_simulator_fn(options, opsums_and_qubit_counts, initial_state)
            elif isinstance(initial_state, np.ndarray):
                evs = ev_simulator_fn(options, opsums_and_qubit_counts, input_vector)
            results.append(evs)

        return results

    def _translate_circuit(
        self,
        circuit: Any,
        translator_fn_name: str,
        qubit_order: ops.QubitOrderOrList,
    ):
        # If the circuit is memoized, reuse the corresponding translated
        # circuit.
        translated_circuit = None
        for original, translated in self._translated_circuits:
            if original == circuit:
                translated_circuit = translated
                break

        if translated_circuit is None:
            translator_fn = getattr(circuit, translator_fn_name)
            translated_circuit = translator_fn(qubit_order)
            self._translated_circuits.append((circuit, translated_circuit))

        return translated_circuit
