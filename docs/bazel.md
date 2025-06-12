# Building with Bazel

qsim provides [Bazel](https://github.com/bazelbuild/bazel) build rules for its
applications and tests. To build and run all tests using Bazel, run the
following command:
```
# AVX and OpenMP are recommended for best performance.
# See "Build configs" section below for more information.
bazel test --config=avx --config=openmp tests:all
```

To run a sample simulation, use the command below. Note that this command
requires the circuit file to be specified both on the command line and in the
`data` field of the `qsim_base` BUILD rule.
```
bazel run --config=avx --config=openmp apps:qsim_base -- -c circuits/circuit_q24
```

## Build configurations

Depending on the optimizers available on your machine, different config flags
(such as `--config=avx`, above) can be set to control which optimizers are
included in a given build or test run.

### Vector arithmetic optimizers

Pick at most one of the following options:

```
# Use AVX instructions for vector arithmetic.
--config=avx

# Use SSE instructions for vector arithmetic.
--config=sse

# Do not use vector arithmetic optimization (default).
--config=basic
```

### Parallelism optimizers

Pick at most one of the following options:

```
# Use OpenMP to run operations in parallel when possible.
--config=openmp

# Do not use OpenMP for parallelism (default).
--config=nopenmp
```

### Memory allocators


[TCMalloc](https://github.com/google/tcmalloc) is a fast, multithreaded
implementation of C's `malloc()` and C++'s `new` operator. It is an independent
open-source library developd by Google. TCMalloc can be used with qsim as an
alternative to the default `malloc()`. Pick at most one of the following
options:

```
# Use TCMalloc for memory allocation.
--config=tcmalloc

# Use malloc for memory allocation (default).
--config=malloc
```

### Additional configuration options

To provide more information when building and testing qsim, you can add the
configuration option `--config=verbose` to any of the `bazel`  commands above.

Other configuration options are described elsewhere in the qsim documentation.
