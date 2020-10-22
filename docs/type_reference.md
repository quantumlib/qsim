# Template naming

This document is meant to clarify the intended usage of common template names
found within qsim. The following table provides example types for each; note
however that there may be other types which are also used for these templates,
and users may define their own alternatives to the examples below so long as
they fulfill the same expectations.

| Name                    |  Example Type                                     |
| ------------------------|---------------------------------------------------|
| Args (parfor.h)         | Arguments to a `Function` object.                 |
| Bitstring               | Alias for `uint64_t`.                             |
| Circuit                 | [`Circuit`](https://github.com/quantumlib/qsim/tree/master/lib/circuit.h)                        |
| Container (util.h)      | A vector of strings, or `Op`'s output type.       |
| Ctype                   | A complex type, e.g. `std::complex<float>`.       |
| For                     | `for` loop abstractions, see below.               |
| ForArgs                 | Arguments for constructing `For` objects, see below. |
| FP (simulator_basic.h)  | Same as `fp_type`.                                |
| fp_type                 | A floating-point type, i.e. `float` or `double`.  |
| Function (parfor.h)     | Any function; args are specified with `Args`.     |
| FuserT                  | [`BasicGateFuser`](https://github.com/quantumlib/qsim/tree/master/lib/fuser_basic.h)             |
| Gate                    | [`Gate`](https://github.com/quantumlib/qsim/tree/master/lib/gate.h)                              |
| GateCirq                | [`GateCirq`](https://github.com/quantumlib/qsim/tree/master/lib/gates_cirq.h)                    |
| GateDef                 | [`GateRX`](https://github.com/quantumlib/qsim/tree/master/lib/gate.h)                            |
| GateQSim                | [`GateQSim`](https://github.com/quantumlib/qsim/tree/master/lib/gates_qsim.h)                    |
| GK / GateKind           | [`GateKind`](https://github.com/quantumlib/qsim/tree/master/lib/gate.h)                          |
| HybridSimulator         | [`HybridSimulator`](https://github.com/quantumlib/qsim/tree/master/lib/hybrid.h)                 |
| IO                      | [`IO`](https://github.com/quantumlib/qsim/tree/master/lib/io.h)                                  |
| IOFile                  | [`IOFile`](https://github.com/quantumlib/qsim/tree/master/lib/io_file.h)                         |
| Matrix                  | A numeric C++ vector representing a matrix.       |
| MeasurementFunc         | [`measure` (in `PrintAmplitudes`)](https://github.com/quantumlib/qsim/tree/master/apps/qsim_base.cc) |
| Op (util.h)             | [`to_int` (in `Options`)](https://github.com/quantumlib/qsim/tree/master/apps/qsim_amplitudes.cc)    |
| ParallelFor             | [`ParallelFor`](https://github.com/quantumlib/qsim/tree/master/lib/parfor.h)                     |
| Params                  | Vector of `fp_type`.                              |
| SequentialFor           | [`SequentialFor`](https://github.com/quantumlib/qsim/tree/master/lib/seqfor.h)                   |
| Simulator               | [`SimulatorAVX`](https://github.com/quantumlib/qsim/tree/master/lib/simulator_avx.h)             |
| State                   | Unique pointer to `fp_type`.                      |
| StateSpace              | [`StateSpace`](https://github.com/quantumlib/qsim/tree/master/lib/statespace.h)                  |
| Stream                  | A valid input for `std::getline()`.               |

## `For` and `ForArgs`

`For` type represents a `for` loop. It is a template parameter of the
`StateSpace*` (lib/statespace*.h) and `Simulator*` (lib/simulator*.h) classes.
`For` objects in these classes are utilized to iterate over quantum state
arrays. `ForArgs` is a variadic template parameter pack of the constructors
of `StateSpace*` and `Simulator*`. It is utilized to pass arguments to the
constructors of `For` objects.

The qsim library provides `ParallelFor` (lib/parfor.h) and `SequentialFor`
(lib/seqfor.h). The user can also use custom `For` types. Examples of usage
follow.

```C++
// ParallelFor(unsigned num_threads) constructor
SimulatorAVX<ParallelFor> simulator(num_qubits, num_threads);
```
```C++
// copy constructor
ParallelFor parallel_for(num_threads);
SimulatorAVX<ParallelFor> simulator(num_qubits, parallel_for);
```
```C++
ParallelFor parallel_for(num_threads);
// const reference to parallel_for in simulator
SimulatorAVX<const ParallelFor&> simulator(num_qubits, parallel_for);
```
In the following, we assume a custom `MyFor` type that has
`MyFor(unsigned num_threads, const Context& context)` and copy constructors.

```C++
// MyFor(unsigned num_threads, const Context& context) constructor
SimulatorAVX<MyFor> simulator(num_qubits, num_threads, context);
```
```C++
// copy constructor
MyFor my_for(num_threads, context);
SimulatorAVX<MyFor> simulator(num_qubits, my_for);
```
```C++
// const reference to my_for in simulator
MyFor my_for(num_threads, context);
SimulatorAVX<const MyFor&> simulator(num_qubits, my_for);
```

## Historical note

For the most part, the usage of templates in qsim (as opposed to interfaces) is
a stylistic choice oriented towards generic programming and policy-based design.
However, there are a small set of cases (most notably in `ParallelFor`) where
the use of templates provides a meaningful performance improvement over other
design patterns.
