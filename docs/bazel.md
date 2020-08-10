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

## Build configs

Depending on the optimizers available on your machine, different config flags
(such as `--config=avx`, above) can be set to control which optimizers are
included in a given build or test run.

Vector arithmetic optimizers (pick one at most):
```
# Use AVX instructions for vector arithmetic.
--config=avx

# Use SSE instructions for vector arithmetic.
--config=sse

# Do not use vector arithmetic optimization (default).
--config=basic
```

Parallelism optimizers (pick one at most):
```
# Use OpenMP to run operations in parallel when possible.
--config=openmp

# Do not use OpenMP for parallelism (default).
--config=nopenmp
```

Memory allocation (pick one at most):
```
# Use tcmalloc for memory allocation.
--config=tcmalloc

# Use malloc for memory allocation (default).
--config=malloc
```
