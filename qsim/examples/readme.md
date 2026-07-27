# Example Circuit Programs

Sample circuit programs demonstrating `qsim` features are located in the `qsim/qsim/examples/` directory.

## 1. Classical Computation: Factorial (`factorial.qs`)

Calculates $n!$ using classical control loops and arithmetic expressions within the [`qsim` circuit description language](../docs/input_format.md). The value of $n$ can be injected dynamically via the command line:

```bash
# Compute 12! (Outputs: 479001600)
./qsim -c examples/factorial.qs -s "n = 12"

```

## 2. Noisy Simulation: Amplitude Damping (`noisy.qs`)

Demonstrates quantum trajectory simulation of a noisy single-qubit circuit subjected to an amplitude damping channel with damping parameter $\gamma$:

```bash
# Sample 100,000 trajectories across 5 parallel workers with gamma = 0.1
./qsim -c examples/noisy.qs -1 100000 -w 5 -s "gamma = 0.1"

```
