# `qsim` Application Reference

The `qsim` application is a command-line simulator for quantum circuits with native support for classical control flow, noisy quantum channels, and parallel repetition sampling. The application source code is located in the `qsim/qsim/` directory.

## Compilation

To compile the `qsim` executable using `make`:

```bash
cd qsim
make

```

Alternatively, to build using Bazel:

```bash
cd qsim
bazel build qsim && cp -p ../bazel-bin/qsim/qsim . && chmod 755 qsim

```

## Command-Line Interface

### Syntax

```bash
./qsim -c <circuit_file> [-s <symbol_defs>] [-0 <rep0>] [-1 <rep1>] \
       [-w <num_workers>] [-t <num_threads_per_worker>] \
       [-f <max_fused_size>] [-v <verbosity>] [-z]

```

### Options & Flags

| Flag | Argument | Description | Default |
| --- | --- | --- | --- |
| **`-c`** | `circuit_file` | Path to the quantum circuit input file. **(Required)** | `""` |
| **`-s`** | `"symbol_defs"` | Space- or semicolon-delimited initial symbol/constant definitions. | `""` |
| **`-0`** | `rep0` | Starting index for trajectory repetition sampling (`seed = r`). | `0` |
| **`-1`** | `rep1` | Upper-bound repetition index (exclusive; total runs = `rep1 - rep0`). | `1` |
| **`-w`** | `num_workers` | Number of parallel worker threads used to execute repetitions concurrently. | `1` |
| **`-t`** | `threads_per_worker` | Number of threads assigned to each worker for state-vector linear algebra. | `1` |
| **`-f`** | `max_fused_size` | Maximum gate fusion size (number of qubits per fused multi-qubit matrix). | `2` |
| **`-v`** | `verbosity` | Logging verbosity level (`0` = silent, `1+` = verbose progress output). | `0` |
| **`-z`** | *(None)* | Enables `Flush-to-Zero` (FTZ) and `Denormals-are-Zeros` (DAZ) in floating-point `MXCSR`. | `false` |

## Detailed Flag Behavior

### Circuit Input (`-c`)

Specifies the file path to the input text file written in the [`qsim` circuit description language](input_format.md).

### Symbol Injection (`-s`)

Injects global constants in the root symbol table prior to circuit parsing. This is useful for parameter sweeps, variable initializations, or setting the circuit qubit count (`nq`):

```bash
./qsim -c circuit.qs -s "nq = 5 gamma = 0.05 theta = 1.57"

```

Types are inferred automatically based on whether the assigned expression is convertible to an integer. Symbol statements can be delimited by spaces or semicolons (`;`).

### Repetitions & Random Seeds (`-0`, `-1`)

Controls the trajectory sampling loop for noisy simulations, classical control branching, and measurement statistics. `qsim` executes trajectories using the following loop logic:

```cpp
for (uint64_t r = rep0; r < rep1; ++r) {
    // The repetition index acts directly as the trajectory RNG seed
    uint64_t seed = r;
    RunTrajectory(circuit, seed);
}

```

* The total number of executed trajectories is **`rep1 - rep0`**.
* Each trajectory index `r` serves directly as the seed for the pseudorandom number generator, ensuring deterministic and reproducible runs.

### Parallel Execution (`-w`, `-t`)

* **`-w <num_workers>`**: Distributes trajectory repetitions across `w` worker threads in parallel.
* **`-t <num_threads_per_worker>`**: Allocates `t` threads per worker for parallel state-vector matrix-vector multiplications.

**Performance Tuning Recommendations:**

* **Small Qubit Counts:** Set `-t 1` (default) and set `-w` to the number of physical CPU cores on your system for maximum throughput across repetitions.
* **Large Qubit Counts:** Set `-t` to the number of physical CPU cores on your system and set `-w 1` (default) to accelerate single-trajectory state-vector operations.

### MXCSR Denormals Control (`-z`)

Enables the SIMD hardware flags `Flush-to-Zero` (FTZ) and `Denormals-are-Zeros` (DAZ). Enabling this option can provide significant speedups during state-vector operations by preventing hardware performance penalties when state amplitudes approach floating-point underflow ($\approx 10^{-38}$).

## Constants

The `qsim` application injects several pre-defined constant identifiers into the symbol table, making them directly accessible within circuit files:

| Identifier | Description | Value |
| --- | --- | --- |
| `pi` | Mathematical constant $\pi$ | `3.14159265358979323846` |
| `rep0` | Starting repetition index | Specified by the `-0` flag |
| `rep1` | Upper-bound repetition index | Specified by the `-1` flag |
| `rid` | Current repetition ID | Dynamic value in the range `[rep0, rep1)` |
| `wid` | Current worker ID | Dynamic value in the range `[0, num_workers)` |

## Usage Examples

### Compute 10 Factorial ($10!$)

```bash
./qsim -c examples/factorial.qs -s "n = 10"

```

### Run Parallel Trajectories

Execute $100,000$ noisy circuit trajectories distributed across 5 parallel worker threads:

```bash
./qsim -c examples/noisy.qs -0 0 -1 100000 -w 5 -s "gamma = 0.1"

```

Sample circuits and additional demonstration files are available in the `qsim/qsim/examples/` directory.
