# Building with Bazel

qsim provides [Bazel](https://github.com/bazelbuild/bazel) build and test rules
for qsim tests and sample applications. The Bazel targets are `tests` and
`apps`; you can combine these with the bazel commands `build`, `test`, and
`run` and configuration flags suitable for your computer hardware architecture
and software environment.

On hardware and software platforms that support them, qsim can be configured to
take advantage of certain hardware optimizations, specifically AVX (a hardware
extension for optimizing vector arithmetic), SSE (streaming SIMD extensions),
and/or OpenMP (a software API for shared-memory parallel programming). By
default, the basic qsim build configuration does _not_ compile in support for
these features. (On some systems such as MacOS on Apple Silicon, they are not
available.) A basic build & test run is obtained using the following command:

```shell
bazel test tests:all
```

As an example of using optimization options, if your computer has support for
AVX and OpenMP, the following command will build and run all the tests with the
appropriate config options to make use of those features:

```shell
bazel test --config=avx --config=openmp tests:all
```

To run a sample simulation, use the command below. Note that this command
requires the `circuit_q24` file to be specified both on the command line and in
the `data` field of the `qsim_base` BUILD rule.

```shell
bazel run --config=avx --config=openmp apps:qsim_base -- -c circuits/circuit_q24
```

## Build configurations

Depending on your computer's hardware architecture and the features available,
different Bazel config flags (such as `--config=avx`, above) can be used to
control which hardware optimizers are included in a given build or test run.

### Vector arithmetic optimizers

Pick at most one of the following options:

```bazel
# Use AVX instructions for vector arithmetic.
--config=avx

# Use SSE instructions for vector arithmetic.
--config=sse

# Do not use vector arithmetic optimization (default).
--config=basic
```

### Parallelism optimizers

Pick at most one of the following options:

```bazel
# Use OpenMP to run operations in parallel when possible.
--config=openmp

# Do not use OpenMP for parallelism (default).
--config=nopenmp
```

### Memory allocators

[TCMalloc](https://github.com/google/tcmalloc) is a fast, multithreaded
implementation of C's `malloc()` and C++'s `new` operator. It is an independent
open-source library developed by Google. TCMalloc can be used with qsim as an
alternative to the default `malloc()`. Pick at most one of the following
options:

```bazel
# Use TCMalloc for memory allocation.
--config=tcmalloc

# Use malloc for memory allocation (default).
--config=malloc
```

### Additional configuration options

To provide more information when building and testing qsim, you can add the
configuration option `--config=verbose` to any of the `bazel`  commands above.

Other configuration options are described elsewhere in the qsim documentation.
