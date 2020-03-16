# The Cirq Interface

This file provides examples of how to use qsim with the
[Cirq](https://github.com/quantumlib/cirq) Python library.

qsim is currently built to work with Cirq version 0.6.0; if you have a later
version of Cirq installed, you may need to downgrade (or run in a virtualenv)
when working with qsim until support for the latest Cirq version is available.


## Setting up

There are two methods for setting up the qsim-Cirq interface on your local
machine: installing directly with `pip`, or compiling from the source code.

Prerequisites:
- [CMake](https://cmake.org/): this is used to compile the C++ qsim libraries.
CMake can be installed with `apt-get install cmake`.
- [Pybind](https://github.com/pybind): this creates Python wrappers for the C++
libraries, and can be installed with `pip3 install pybind11`.
- [Cirq](https://cirq.readthedocs.io/en/stable/install.html).

### Installing with pip

qsim can be installed with `pip3 install qsimcirq`. Alternatives (such as
installing with Docker) can be found in the
[installation doc](/docs/install_qsimcirq.md).

### Compiling qsimcirq

1. Clone the qsim repository to your machine, and navigate to the top-level
`qsim` directory:
```
git clone git@github.com:quantumlib/qsim.git
cd qsim
```

2. Compile qsim using the top-level Makefile: `make`. By default, this will use
Pybind to generate a static library with file extension `.so` in the `qsimcirq`
directory.

3. To verify successful compilation, run the Python tests:
```
make run-py-tests
```
This will run [qsim_test](/qsimcirq_tests/qsim_test.py), which calls the Python
binary directly, and [qsimcirq_test](/qsimcirq_tests/qsimcirq_test.py), which
invokes qsim through the qsim-Cirq interface.

## Interface design and operations

The purpose of this interface is to provide a performant simulator for quantum
circuits defined in Cirq. 

### Classes

The interface includes QSimSimulator and QSimhSimulator which communicates
through a Pybind11 interface with qsim. The simulator accepts only QSimCircuit,
which is effectively a specialized kind of `cirq.Circuit`.

Architectural constraints such as permitting only qsim supported
gate sets, and different circuit validations are performed by the
QSimCircuit.

### Usage procedure

A QSimCircuit can be created from a Cirq circuit.
```
my_circuit = cirq.Circuit()
qsim_circuit = qsimcirq.QSimCircuit(cirq_circuit=my_circuit)
```

This circuit can then be simulated using either QSimSimulator or
QSimhSimulator, depending on the output required:

#### QSimSimulator

QSimSimulator uses a Schrödinger full state-vector simulator, suitable for
acquiring the complete state of a reasonably-sized circuit (~35 qubits):
```
my_sim = qsimcirq.QSimSimulator()
myres = my_sim.simulate(program = my_sim_circuit)
```

Alternatively, by using the `compute_amplitudes` method QSimSimulator can
produce amplitudes for specific output bitstrings:
```
my_sim = qsimcirq.QSimSimulator()
myres = my_sim.compute_amplitudes(program = my_sim_circuit,
                                  bitstrings=['00', '01', '10', '11'])
```
In the above example, the simulation is performed for the specified bitstrings
of length 2. All the bitstring lengths should be equal to the number of qubits
in `qsim_circuit`. Otherwise, BitstringsFromStream will raise an error.

#### QSimhSimulator

QSimhSimulator uses a hybrid Schrödinger-Feynman simulator. This limits it to
returning amplitudes for specific output bitstrings, but raises its upper
bound on number of qubits simulated (50+ qubits, depending on depth).

To acquire amplitudes for all output bitstrings of length 2:
```
qsimh_options = {
    'k': [0],
    'w': 0,
    'p': 1,
    'r': 1
}
my_sim = qsimcirq.QSimhSimulator(qsimh_options)
myres = my_sim.compute_amplitudes(program = my_sim_circuit,
                                  bitstrings=['00', '01', '10', '11'])
```


## Use qsim from Python without Cirq

It is possible to call the qsim binaries from Python without using Cirq.
To see this in action, run the [qsim_test](interfaces/tests/qsim_test.py):
```
python3 -m pytest interfaces/tests/qsim_test.py
```


## Experimental features

This version of qsim includes preliminary support for gate decompositions and
parametrized operations. Users relying on these features should have a good
working knowledge of Cirq, as some non-trivial work is required to set them up.

### Gate decompositions

The QSimCircuit is capable of decomposing arbitrary Cirq gates to the
elementary gate set of qsim. However, if the user does not specify valid
decompositions, the QSimCircuit composition will raise exceptions.

The constructor takes the argument `allow_decomposition` which is `False` by
default. If set `True`, the gates from the original circuit will decomposed,
using their ` _decompose_()` contract. For example, the
[cirq.CNOT is decomposed into](https://github.com/quantumlib/Cirq/blob/49b2f193ad99ce6770831330c19963bfa5c66f19/cirq/ops/common_gates.py#L829):
```
yield YPowGate(exponent=-0.5).on(t)
yield CZ(c, t)**self._exponent
yield YPowGate(exponent=0.5).on(t)
```

### Parametrized circuits

In theory, QSimCircuit objects can contain
[parameterized gates](https://cirq.readthedocs.io/en/stable/tutorial.html#parameterizing-the-ansatz)
which have values assigned by Cirq's `ParamResolver`. However, this
functionality has not been tested extensively.
