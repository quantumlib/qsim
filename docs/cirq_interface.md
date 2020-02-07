# The Cirq Interface

HOWTO use qsim from Google [Cirq](https://github.com/quantumlib/cirq).
* Note: For the moment, the Cirq-qsim integration is experimental code.

The currently supported version of Cirq is 0.5.0, and newer stable
versions will be supported, once they are available. Therefore, the
prerequisites are:
- Cirq 0.5.0
- qsim C++ code (the code from this repo)

This file is an example of how to use the qsim Python interface with Cirq.


## Setting up

1. Cirq needs to be [installed on your machine](https://cirq.readthedocs.io/en/stable/install.html)

2. Compile qsim with Pybind support (see [make.sh](/cirq_interface/cpp/make.sh)). The `cpp` folder should contain a shared library that is available as module to Python.

3. Ensure that compiled qsim module load correctly by adding `cpp` folder to PYTHONPATH:
`export PYTHONPATH=$PYTHONPATH:<path to cpp folder>`

4. Run the examples given in [examples](/cirq_interface/examples) folder.
`python3 cirq_qsim_example.py`


## Interface design and operations

The goal is to simulate circuits using native Cirq objects.

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
```

At this point, `my_circuit` is a cirq.Circuit, and can be used to construct a
QSimCircuit
```
my_sim_circuit = qcirc.QSimCircuit(cirq_circuit = my_circuit, device = my_device)
```

It is possible to specify if gate decompositions should be performed or not (see Notes).

Finally, the QSimCircuit can be simulated using QSimSimulator and QSimhSimulator classes.

QSimSimulator supports full state vector as well as the specific bitstrings simulation.
```
my_sim = qsim.QSimSimulator()
myres = my_sim.compute_amplitudes(program = my_sim_circuit,
                                  bitstrings=['00', '01', '10', '11'])
```
In the above example, the simulation is performed for the specified bitstrings of length 2. All the bitstring lengths should be equal to the number of qubits in my_sim_circuits. Otherwise, BitstringsFromStream will raise an error.

```
my_sim = qsim.QSimSimulator()
myres = my_sim.simulate(program = my_sim_circuit)
```

QSimhSimulator supports specific bitstrings simulation.
```
qsimh_options = {
    'k': [0],
    'w': 0,
    'p': 1,
    'r': 1
}
my_sim = qsim.QSimhSimulator(qsimh_options)
myres = my_sim.compute_amplitudes(program = my_sim_circuit,
                                  bitstrings=['00', '01', '10', '11'])
```


## Notes

This version includes preliminary support for gate decompositions and
parametrized operations. Users relying on them should know how Cirq works behind
the scenes, in order to write the code that is still required.

### Gate decompositions
* TODO: Enable the full functionality.

The QSimCircuit is capable of decomposing arbitrary Cirq gates to the
elementary gate set of qsim. However, if the user does not specify valid
decompositions, the QSimCircuit composition will raise exceptions.

The constructor takes the argument `allow_decomposition` which is `False` by
default. If set `True`, the gates from the original circuit will decomposed,
using their ` _decompose_()` contract. For example, the [cirq.CNOT is decomposed
into](https://github.com/quantumlib/Cirq/blob/49b2f193ad99ce6770831330c19963bfa5c66f19/cirq/ops/common_gates.py#L829):
```
yield YPowGate(exponent=-0.5).on(t)
yield CZ(c, t)**self._exponent
yield YPowGate(exponent=0.5).on(t)
```

### Parametrized circuits
* TODO: Enable the full functionality.

If needed, file an issue.

### Use qsim from Python without Cirq

This is possible by using the Pybind11 interface, which, currently, can be used
as shown in [example.py](/cirq_interface/examples/example.py).
