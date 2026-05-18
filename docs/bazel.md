# Building with Bazel

qsim provides [Bazel](https://github.com/bazelbuild/bazel) build and test rules
for qsim tests and sample applications. The Bazel targets are `tests` and
`apps`; you can combine these with the bazel commands `build`, `test`, and
`run` and configuration flags suitable for your computer hardware architecture
and software environment.

On hardware and software platforms that support them, qsim can be configured to
take advantage of certain CPU hardware extensions for optimizing operations.
Specifically, qsim can take advantage of AVX (Advanced Vector Extensions), SSE
(Streaming SIMD Extensions), BMI (Bit Manipulation Instructions), and/or OpenMP
(shared-memory parallel programming).

By default, the basic qsim build configuration does _not_ include these
optimizations. A basic build and test run is obtained using the following
command:

```shell
bazel test tests:all
```

Conversely, if you want to get all optimizations possible, you can use this:

```shell
bazel test --config=native --config=openmp tests:all
```

## Build configurations

qsim uses an additive, host-aware configuration model. Multiple hardware
features can be enabled simultaneously, and Bazel will automatically verify
that your host CPU supports the requested instruction sets.

### Vector arithmetic and bit manipulation

You can combine one or more of the following options. Bazel will fail fast if
the host hardware does not support a requested feature.

```bazel
# Use AVX2 and AVX instructions.
--config=avx2

# Use AVX512 instructions (implies AVX2/AVX).
--config=avx512

# Use AVX instructions (includes AVX2 and AVX512).
--config=avx

# Use SSE 4.1 instructions.
--config=sse

# Use BMI2 instructions.
--config=bmi

# Automatically detect and use the best instruction set for the host.
--config=native

# Do not use AVX, SSE, or BMI optimizations (default).
--config=basic
```

For example, if your computer supports AVX2, SSE, and BMI, you can enable all
of them:

```shell
bazel test --config=avx2 --config=sse --config=bmi tests:all
```

For another example, if you want to let the build system choose all the
optimizations supported on your host computer, you can use this:

```shell
bazel test --config=native tests:all
```

### Parallelism optimizers

qsim can take advantage of [OpenMP]( https://en.wikipedia.org/wiki/OpenMP)
(Open Multi-Processing), an industry-standard API for shared-memory parallel
programming. You can enable the use of OpenMP this way:

```bazel
# Use OpenMP to run operations in parallel when possible.
--config=openmp
```

To explicitly disable the use of OpenMP, you can use `--config=nopenmp` or not
use the OpenMP option at all (which is the default).

### Memory allocators

[TCMalloc](https://github.com/google/tcmalloc) can be used as a faster
alternative to the default `malloc()`. Consult the TCMalloc documentation for
information about how to install it. If it is available on your system, you
can tell Bazel to build qsim with it like this:

```bazel
# Use TCMalloc for memory allocation.
--config=tcmalloc

# Use malloc for memory allocation (default).
--config=malloc
```

### Diagnostics and Verbosity

To see the active compiler flags and detected host CPU features, add the
`--config=verbose` configuration to any of the other options. For example,

```shell
bazel build --config=verbose --config=native tests:all
```

You can also add `--config=verbose` to any `test` or `run` command for
detailed build information.

---

For more details on running sample simulations, see the documentation for
individual apps in the `apps/` directory.
