# The Cirq Interface

This file provides examples of how to use qsim with the
[Cirq](https://github.com/quantumlib/cirq) Python library.

qsim is kept up-to-date with the latest version of Cirq. If you experience
compatibility issues, please file an issue in the qsim or Cirq repository
as appropriate.


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
[installation doc](./install_qsimcirq.md).

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
This will run
[qsimcirq_test](https://github.com/quantumlib/qsim/blob/master/qsimcirq_tests/qsimcirq_test.py),
which invokes qsim through the qsim-Cirq interface.

## Interface design and operations

The purpose of this interface is to provide a performant simulator for quantum
circuits defined in Cirq. 

### Classes

The interface includes QSimSimulator and QSimhSimulator which communicate
through a Pybind11 interface with qsim. The simulator accepts `cirq.Circuit`
objects, which it wraps as `QSimCircuit` to enforce architectural constraints
(such as decomposing to qsim-supported gate sets).

### Usage procedure

We begin by defining a Cirq circuit which we want to simulate.

```
my_circuit = cirq.Circuit()
```

This circuit can then be simulated using either `QSimSimulator` or
`QSimhSimulator`, depending on the desired output.

#### QSimSimulator

`QSimSimulator` uses a Schrödinger full state-vector simulator, suitable for
acquiring the complete state of a reasonably-sized circuit (~25 qubits on an
average PC, or up to 40 qubits on high-performance VMs).
Options for the simulator, including number of threads and verbosity, can be
set with the `qsim_options` field using the `qsim_base` flag format defined in
the [usage docs](./usage.md).

```
qsim_options = {'t': 8, 'v': 0}
my_sim = qsimcirq.QSimSimulator(qsim_options)
myres = my_sim.simulate(program=my_circuit)
```

Alternatively, by using the `compute_amplitudes` method `QSimSimulator` can
produce amplitudes for specific output bitstrings:
```
my_sim = qsimcirq.QSimSimulator()
myres = my_sim.compute_amplitudes(program=my_circuit,
                                  bitstrings=[0b00, 0b01, 0b10, 0b11])
```
In the above example, the simulation is performed for the specified bitstrings
of length 2. All the bitstring lengths should be equal to the number of qubits
in `qsim_circuit`. Otherwise, BitstringsFromStream will raise an error.

Finally, to retrieve sample measurements the `run` method can be used. This requires
the circuit to have measurements to sample from, else an error will be raised.
```
my_sim = qsimcirq.QSimSimulator()
myres = my_sim.run(program=my_circuit)
```

This method may be more efficient if the final state vector is very large, as
it only returns a bitstring produced by sampling from the final state. It also
allows intermediate measurements to be applied to the circuit.

Note that requesting multiple repetitions with the `run` method will execute
the circuit once for each repetition unless all measurements are terminal. This
ensures that nondeterminism from intermediate measurements is properly
reflected in the results.

#### QSimhSimulator

`QSimhSimulator` uses a hybrid Schrödinger-Feynman simulator. This limits it to
returning amplitudes for specific output bitstrings, but raises its upper
bound on number of qubits simulated (50+ qubits, depending on depth).

To acquire amplitudes for all output bitstrings of length 2:
```
qsimh_options = {
    'k': [0],
    'w': 0,
    'p': 0,
    'r': 2
}
my_sim = qsimcirq.QSimhSimulator(qsimh_options)
myres = my_sim.compute_amplitudes(program=my_circuit,
                                  bitstrings=[0b00, 0b01, 0b10, 0b11])
```

As with `QSimSimulator`, the options follow the flag format for `qsimh_base`
outlined in the [usage docs](./usage.md).

## Additional features

The qsim-Cirq interface provides basic support for gate decomposition and
circuit parameterization.

### Gate decompositions

Circuits received by qsimcirq are automatically decomposed into the qsim
gate set if possible. This uses the Cirq `decompose` operation. Gates with no
decomposition to the qsim gate set will instead attempt to be parsed as raw
matrices, if one is specified.

### Parametrized circuits

QSimCircuit objects can also contain
[parameterized gates](https://cirq.readthedocs.io/en/stable/docs/tutorials/basics.html#Using-parameter-sweeps)
which have values assigned by Cirq's `ParamResolver`. See the link above for
details on how to use this feature.
