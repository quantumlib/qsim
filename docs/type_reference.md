# Template naming

This document is meant to clarify the intended usage of common template names
found within qsim. The following table provides example types for each; note
however that there may be other types which are also used for these templates,
and users may define their own alternatives to the examples below so long as
they fulfill the same expectations.

| Name                    |  Example Type                                     |
| ------------------------|---------------------------------------------------|
| Args (parfor.h)         | Arguments to a `Function` object.                 |
| Array1 / Array2         | A numeric C++ array representing a matrix.        |
| Bitstring               | Alias for `uint64_t`.                             |
| Circuit                 | [`Circuit`](lib/circuit.h)                        |
| Container (util.h)      | A vector of strings, or `Op`'s output type.       |
| Ctype                   | A complex type, e.g. `std::complex<float>`.       |
| FP (simulator_basic.h)  | Same as `fp_type`.                                |
| fp_type                 | A floating-point type, i.e. `float` or `double`.  |
| Function (parfor.h)     | Any function; args are specified with `Args`.     |
| FuserT                  | [`BasicGateFuser`](lib/fuser_basic.h)             |
| Gate                    | [`Gate`](lib/gate.h)                              |
| GateDef                 | [`GateRX`](lib/gate.h)                            |
| GateQSim                | [`GateQSim`](lib/gates_qsim.h)                    |
| GK / GateKind           | [`GateKind`](lib/gate.h)                          |
| HybridSimulator         | [`HybridSimulator`](lib/hybrid.h)                 |
| IO                      | [`IO`](lib/io.h)                                  |
| IOFile                  | [`IOFile`](lib/io_file.h)                         |
| Matrix                  | Same as `Array(1|2)`.                             |
| MeasurementFunc         | [`measure` (in `PrintAmplitudes`)](apps/qsim_base.cc) |
| Op (util.h)             | [`to_int` (in `Options`)](apps/qsim_amplitudes.cc)    |
| ParallelFor             | [`ParallelFor`](lib/parfor.h)                     |
| Params                  | Vector of `fp_type`.                              |
| Simulator               | [`SimulatorAVX`](lib/simulator_avx.h)             |
| State                   | Unique pointer to `fp_type`.                      |
| StateSpace              | [`StateSpace`](lib/statespace.h)                  |
| Stream                  | A valid input for `std::getline()`.               |

## Historical note

For the most part, the usage of templates in qsim (as opposed to interfaces) is
a stylistic choice oriented towards generic programming and policy-based design.
However, there are a small set of cases (most notably in `ParallelFor`) where
the use of templates provides a meaningful performance improvement over other
design patterns.
