# Building with Bazel

<!-- TODO: expand to explain other Bazel build options. -->
qsim provides [Bazel](https://github.com/bazelbuild/bazel) build rules for its
applications and tests. To build and run all tests using Bazel, run the
following command:
```
bazel test --config=avx tests:all
```

To run a sample simulation, use the command below. Note that this command
requires the circuit file to be specified both on the command line and in the
`data` field of the `qsim_base` BUILD rule.
```
bazel run --config=avx apps:qsim_base -- -c circuits/circuit_q24
```