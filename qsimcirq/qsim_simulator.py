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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cirq

import numpy as np

from . import qsim, qsim_gpu, qsim_custatevec
import qsimcirq.qsim_circuit as qsimc


# This should probably live in Cirq...
# TODO: update to support CircuitOperations.
def _needs_trajectories(circuit: cirq.Circuit) -> bool:
    """Checks if the circuit requires trajectory simulation."""
    for op in circuit.all_operations():
        test_op = (
            op
            if not cirq.is_parameterized(op)
            else cirq.resolve_parameters(
                op, {param: 1 for param in cirq.parameter_names(op)}
            )
        )
        if not (cirq.is_measurement(test_op) or cirq.has_unitary(test_op)):
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
        gpu_state_threads: number of threads per CUDA block to use for the GPU
            StateSpace. This must be a power of 2 in the range [32, 1024].
        gpu_data_blocks: number of data blocks to use for the GPU StateSpace.
            Below 16 data blocks, performance is noticeably reduced.
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
            "gsst": self.gpu_state_threads,
            "gdb": self.gpu_data_blocks,
            "v": self.verbosity,
            "z": self.denormals_are_zeros,
        }


@dataclass
class MeasInfo:
    """Info about each measure operation in the circuit being simulated.

    Attributes:
        key: The measurement key.
        idx: The "instance" of a possibly-repeated measurement key.
        invert_mask: True for any measurement bits that should be inverted.
        start: Start index in qsim's output array for this measurement.
        end: End index (non-inclusive) in qsim's output array.
    """

    key: str
    idx: int
    invert_mask: Tuple[bool, ...]
    start: int
    end: int


class QSimSimulator(
    cirq.SimulatesSamples,
    cirq.SimulatesAmplitudes,
    cirq.SimulatesFinalState[cirq.StateVectorTrialResult],
    cirq.SimulatesExpectationValues,
):
    def __init__(
        self,
        qsim_options: Union[None, Dict, QSimOptions] = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        noise: cirq.NOISE_MODEL_LIKE = None,
        circuit_memoization_size: int = 0,
    ):
        """Creates a new QSimSimulator using the given options and seed.

        Args:
            qsim_options: An options dict or QSimOptions object with options
                to use for all circuits run using this simulator. See the
                QSimOptions class for details.
            seed: A random state or seed object, as defined in cirq.value.
            noise: A cirq.NoiseModel to apply to all circuits simulated with
                this simulator.
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
        self._prng = cirq.value.parse_random_state(seed)
        self.qsim_options = QSimOptions().as_dict()
        self.qsim_options.update(qsim_options)
        self.noise = cirq.NoiseModel.from_noise_model_like(noise)

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

        # Deque of (
        #   <original cirq circuit>,
        #   <translated qsim circuit>,
        #   <moment_gate_indices>
        # ) tuples.
        self._translated_circuits = deque(maxlen=circuit_memoization_size)

    def get_seed(self):
        # Limit seed size to 32-bit integer for C++ conversion.
        return self._prng.randint(2**31 - 1)

    def _run(
        self,
        circuit: cirq.Circuit,
        param_resolver: cirq.ParamResolver,
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.

        Returns:
            A dictionary from measurement gate key to measurement
            results.
        """
        param_resolver = param_resolver or cirq.ParamResolver({})
        solved_circuit = cirq.resolve_parameters(circuit, param_resolver)

        return self._sample_measure_results(solved_circuit, repetitions)

    def _sample_measure_results(
        self,
        program: cirq.Circuit,
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

        # Add noise to the circuit if a noise model was provided.
        all_qubits = program.all_qubits()
        program = qsimc.QSimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        # Compute indices of measured qubits
        ordered_qubits = cirq.QubitOrder.DEFAULT.order_for(all_qubits)
        num_qubits = len(ordered_qubits)

        qubit_map = {qubit: index for index, qubit in enumerate(ordered_qubits)}

        # Compute:
        # - number of qubits for each measurement key.
        # - measurement ops for each measurement key.
        # - measurement info for each measurement.
        # - total number of measured bits.
        measurement_ops = [
            op
            for _, op, _ in program.findall_operations_with_gate_type(
                cirq.MeasurementGate
            )
        ]
        num_qubits_by_key: Dict[str, int] = {}
        meas_ops: Dict[str, List[cirq.GateOperation]] = {}
        meas_infos: List[MeasInfo] = []
        num_bits = 0
        for op in measurement_ops:
            gate = op.gate
            key = cirq.measurement_key_name(gate)
            meas_ops.setdefault(key, [])
            i = len(meas_ops[key])
            meas_ops[key].append(op)
            n = len(op.qubits)
            if key in num_qubits_by_key:
                if n != num_qubits_by_key[key]:
                    raise ValueError(
                        f"repeated key {key!r} with different numbers of qubits: "
                        f"{num_qubits_by_key[key]} != {n}"
                    )
            else:
                num_qubits_by_key[key] = n
            meas_infos.append(
                MeasInfo(
                    key=key,
                    idx=i,
                    invert_mask=gate.full_invert_mask(),
                    start=num_bits,
                    end=num_bits + n,
                )
            )
            num_bits += n

        # Set qsim options
        options = {**self.qsim_options}

        results = {
            key: np.ndarray(shape=(repetitions, len(meas_ops[key]), n), dtype=int)
            for key, n in num_qubits_by_key.items()
        }

        noisy = _needs_trajectories(program)
        if not noisy and program.are_all_measurements_terminal() and repetitions > 1:
            # Measurements must be replaced with identity gates to sample properly.
            # Simply removing them may omit qubits from the circuit.
            for i in range(len(program.moments)):
                program.moments[i] = cirq.Moment(
                    op
                    if not isinstance(op.gate, cirq.MeasurementGate)
                    else [cirq.IdentityGate(1).on(q) for q in op.qubits]
                    for op in program.moments[i]
                )
            translator_fn_name = "translate_cirq_to_qsim"
            options["c"], _ = self._translate_circuit(
                program,
                translator_fn_name,
                cirq.QubitOrder.DEFAULT,
            )
            options["s"] = self.get_seed()
            raw_results = self._sim_module.qsim_sample_final(options, repetitions)
            full_results = np.array(
                [
                    [bool(result & (1 << q)) for q in reversed(range(num_qubits))]
                    for result in raw_results
                ]
            )

            for key, oplist in meas_ops.items():
                for i, op in enumerate(oplist):
                    meas_indices = [qubit_map[qubit] for qubit in op.qubits]
                    invert_mask = op.gate.full_invert_mask()
                    # Apply invert mask to re-ordered results
                    results[key][:, i, :] = full_results[:, meas_indices] ^ invert_mask

        else:
            if noisy:
                translator_fn_name = "translate_cirq_to_qtrajectory"
                sampler_fn = self._sim_module.qtrajectory_sample
            else:
                translator_fn_name = "translate_cirq_to_qsim"
                sampler_fn = self._sim_module.qsim_sample

            options["c"], _ = self._translate_circuit(
                program,
                translator_fn_name,
                cirq.QubitOrder.DEFAULT,
            )
            measurements = np.empty(shape=(repetitions, num_bits), dtype=int)
            for i in range(repetitions):
                options["s"] = self.get_seed()
                measurements[i] = sampler_fn(options)

            for m in meas_infos:
                results[m.key][:, m.idx, :] = (
                    measurements[:, m.start : m.end] ^ m.invert_mask
                )

        return results

    def compute_amplitudes_sweep_iter(
        self,
        program: cirq.Circuit,
        bitstrings: Sequence[int],
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
    ) -> Iterator[Sequence[complex]]:
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

        Yields:
            Amplitudes.
        """

        # Add noise to the circuit if a noise model was provided.
        all_qubits = program.all_qubits()
        program = qsimc.QSimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        # qsim numbers qubits in reverse order from cirq
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        num_qubits = len(cirq_order)
        bitstrings = [
            format(bitstring, "b").zfill(num_qubits)[::-1] for bitstring in bitstrings
        ]

        options = {"i": "\n".join(bitstrings)}
        options.update(self.qsim_options)

        param_resolvers = cirq.to_resolvers(params)

        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            simulator_fn = self._sim_module.qtrajectory_simulate
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            simulator_fn = self._sim_module.qsim_simulate

        for prs in param_resolvers:
            solved_circuit = cirq.resolve_parameters(program, prs)
            options["c"], _ = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()
            yield simulator_fn(options)

    def _simulate_impl(
        self,
        program: cirq.Circuit,
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Optional[Union[int, np.ndarray]] = None,
    ) -> Iterator[Tuple[cirq.ParamResolver, np.ndarray, Sequence[int]]]:
        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an int or state vector.")

        # Add noise to the circuit if a noise model was provided.
        all_qubits = program.all_qubits()
        program = qsimc.QSimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.qsim_options)

        param_resolvers = cirq.to_resolvers(params)
        # qsim numbers qubits in reverse order from cirq
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        qsim_order = list(reversed(cirq_order))
        num_qubits = len(qsim_order)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64.")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2**num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )

        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            fullstate_simulator_fn = self._sim_module.qtrajectory_simulate_fullstate
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            fullstate_simulator_fn = self._sim_module.qsim_simulate_fullstate

        for prs in param_resolvers:
            solved_circuit = cirq.resolve_parameters(program, prs)

            options["c"], _ = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()

            if isinstance(initial_state, int):
                qsim_state = fullstate_simulator_fn(options, initial_state)
            elif isinstance(initial_state, np.ndarray):
                qsim_state = fullstate_simulator_fn(options, input_vector)
            assert qsim_state.dtype == np.float32
            assert qsim_state.ndim == 1

            yield prs, qsim_state.view(np.complex64), cirq_order

    def simulate_into_1d_array(
        self,
        program: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolverOrSimilarType = None,
        qubit_order: cirq.QubitOrderOrList = cirq.ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Tuple[cirq.ParamResolver, np.ndarray, Sequence[int]]:
        """Same as simulate() but returns raw simulation result without wrapping it.

            The returned result is not wrapped in a StateVectorTrialResult but can be used
            to create a StateVectorTrialResult.

        Returns:
            Tuple of (param resolver, final state, qubit order)
        """
        params = cirq.study.ParamResolver(param_resolver)
        return next(self._simulate_impl(program, params, qubit_order, initial_state))

    def simulate_sweep_iter(
        self,
        program: cirq.Circuit,
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Optional[Union[int, np.ndarray]] = None,
    ) -> Iterator[cirq.StateVectorTrialResult]:
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
            Iterator over SimulationTrialResults for this run, one for each
            possible parameter resolver.

        Raises:
            TypeError: if an invalid initial_state is provided.
        """

        for prs, state_vector, cirq_order in self._simulate_impl(
            program, params, qubit_order, initial_state
        ):
            final_state = cirq.StateVectorSimulationState(
                initial_state=state_vector, qubits=cirq_order
            )
            # create result for this parameter
            # TODO: We need to support measurements.
            yield cirq.StateVectorTrialResult(
                params=prs, measurements={}, final_simulator_state=final_state
            )

    def simulate_expectation_values_sweep_iter(
        self,
        program: cirq.Circuit,
        observables: Union[cirq.PauliSumLike, List[cirq.PauliSumLike]],
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> Iterator[List[float]]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            params: Parameters to run with the program.
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

        Yields:
            Lists of expectation values, with the value at index `n`
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
        psumlist = [cirq.PauliSum.wrap(pslike) for pslike in observables]

        all_qubits = program.all_qubits()
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
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

        # Add noise to the circuit if a noise model was provided.
        program = qsimc.QSimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.qsim_options)

        param_resolvers = cirq.to_resolvers(params)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64.")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2**num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )

        if _needs_trajectories(program):
            translator_fn_name = "translate_cirq_to_qtrajectory"
            ev_simulator_fn = self._sim_module.qtrajectory_simulate_expectation_values
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            ev_simulator_fn = self._sim_module.qsim_simulate_expectation_values

        for prs in param_resolvers:
            solved_circuit = cirq.resolve_parameters(program, prs)
            options["c"], _ = self._translate_circuit(
                solved_circuit,
                translator_fn_name,
                cirq_order,
            )
            options["s"] = self.get_seed()

            if isinstance(initial_state, int):
                evs = ev_simulator_fn(options, opsums_and_qubit_counts, initial_state)
            elif isinstance(initial_state, np.ndarray):
                evs = ev_simulator_fn(options, opsums_and_qubit_counts, input_vector)
            yield evs

    def simulate_moment_expectation_values(
        self,
        program: cirq.Circuit,
        indexed_observables: Union[
            Dict[int, Union[cirq.PauliSumLike, List[cirq.PauliSumLike]]],
            cirq.PauliSumLike,
            List[cirq.PauliSumLike],
        ],
        param_resolver: cirq.ParamResolver,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List[List[float]]:
        """Calculates expectation values at each moment of a circuit.

        Args:
            program: The circuit to simulate.
            indexed_observables: A map of moment indices to an observable
                or list of observables to calculate after that moment. As a
                convenience, users can instead pass in a single observable
                or observable list to calculate after ALL moments.
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
            A list of expectation values for each moment m in the circuit,
            where value `n` corresponds to `indexed_observables[m][n]`.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False. (Note: We cannot test this
            until Cirq's `are_any_measurements_terminal` is released.)
        """
        if not isinstance(indexed_observables, Dict):
            if not isinstance(indexed_observables, List):
                indexed_observables = [
                    (i, [indexed_observables]) for i, _ in enumerate(program)
                ]
            else:
                indexed_observables = [
                    (i, indexed_observables) for i, _ in enumerate(program)
                ]
        else:
            indexed_observables = [
                (i, obs) if isinstance(obs, List) else (i, [obs])
                for i, obs in indexed_observables.items()
            ]
        indexed_observables.sort(key=lambda x: x[0])
        psum_pairs = [
            (i, [cirq.PauliSum.wrap(pslike) for pslike in obs_list])
            for i, obs_list in indexed_observables
        ]

        all_qubits = program.all_qubits()
        cirq_order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(all_qubits)
        qsim_order = list(reversed(cirq_order))
        num_qubits = len(qsim_order)
        qubit_map = {qubit: index for index, qubit in enumerate(qsim_order)}

        opsums_and_qcount_map = {}
        for i, psumlist in psum_pairs:
            opsums_and_qcount_map[i] = []
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
                opsums_and_qcount_map[i].append((opsum, len(opsum_qubits)))

        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (int, np.ndarray)):
            raise TypeError("initial_state must be an int or state vector.")

        # Add noise to the circuit if a noise model was provided.
        program = qsimc.QSimCircuit(
            self.noise.noisy_moments(program, sorted(all_qubits))
            if self.noise is not cirq.NO_NOISE
            else program,
        )

        options = {}
        options.update(self.qsim_options)

        param_resolver = cirq.to_resolvers(param_resolver)
        if isinstance(initial_state, np.ndarray):
            if initial_state.dtype != np.complex64:
                raise TypeError(f"initial_state vector must have dtype np.complex64.")
            input_vector = initial_state.view(np.float32)
            if len(input_vector) != 2**num_qubits * 2:
                raise ValueError(
                    f"initial_state vector size must match number of qubits."
                    f"Expected: {2**num_qubits * 2} Received: {len(input_vector)}"
                )

        is_noisy = _needs_trajectories(program)
        if is_noisy:
            translator_fn_name = "translate_cirq_to_qtrajectory"
            ev_simulator_fn = (
                self._sim_module.qtrajectory_simulate_moment_expectation_values
            )
        else:
            translator_fn_name = "translate_cirq_to_qsim"
            ev_simulator_fn = self._sim_module.qsim_simulate_moment_expectation_values

        solved_circuit = cirq.resolve_parameters(program, param_resolver)
        options["c"], opsum_reindex = self._translate_circuit(
            solved_circuit,
            translator_fn_name,
            cirq_order,
        )
        opsums_and_qubit_counts = []
        for m, opsum_qc in opsums_and_qcount_map.items():
            pair = (opsum_reindex[m], opsum_qc)
            opsums_and_qubit_counts.append(pair)
        options["s"] = self.get_seed()

        if isinstance(initial_state, int):
            return ev_simulator_fn(options, opsums_and_qubit_counts, initial_state)
        elif isinstance(initial_state, np.ndarray):
            return ev_simulator_fn(options, opsums_and_qubit_counts, input_vector)

    def _translate_circuit(
        self,
        circuit: Any,
        translator_fn_name: str,
        qubit_order: cirq.QubitOrderOrList,
    ):
        # If the circuit is memoized, reuse the corresponding translated circuit.
        for original, translated, moment_indices in self._translated_circuits:
            if original == circuit:
                return translated, moment_indices

        translator_fn = getattr(circuit, translator_fn_name)
        translated, moment_indices = translator_fn(qubit_order)
        self._translated_circuits.append((circuit, translated, moment_indices))

        return translated, moment_indices
