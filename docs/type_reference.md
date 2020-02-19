# Template naming

Under certain circumstances, C++ interfaces can introduce time-cost overhead
when compared with inline template functions. Since some of the functions in
qsim are called with very high frequency, templates are preferred to keep
runtime to a minimum.

This has the unfortunate side effect of obscuring which types are relevant in
each function; in order to alleviate this difficulty, the following table
provides example types for each of the common template names used in qsim.

**Note:** these types are intended to represent the expectations qsim places on
each template type. There may be other types which are also used for these
templates, and users may define their own alternatives to the examples below so
long as they fulfill the same expectations.

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
| GateDef                 | [`GateRX`](lib/gatedef.h)                         |
| HybridSimulator         | [`HybridSimulator`](lib/hybrid.h)                 |
| IO                      | [`IO`](lib/io.h)                                  |
| Matrix                  | Same as `Array(1|2)`.                             |
| MeasurementFunc         | [`measure` (in `PrintAmplitudes`)](apps/qsim1.cc) |
| Op (util.h)             | [`to_int` (in `Options`)](apps/qsim3.cc)          |
| ParallelFor             | [`ParallelFor`](lib/parfor.h)                     |
| Params                  | Vector of `fp_type`.                              |
| Simulator               | [`SimulatorAVX`](lib/simulator_avx.h)             |
| State                   | Unique pointer to `fp_type`.                      |
| StateSpace              | [`StateSpace`](lib/statespace.h)                  |
| Stream                  | A valid input for `std::getline()`.               |
