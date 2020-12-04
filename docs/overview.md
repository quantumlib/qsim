# qsim and qsimh

qsim and qsimh are a collection of C++ libraries for quantum circuit
simulation. These libraries provide powerful, low-cost tools for
researchers to test quantum algorithms before running on quantum hardware.

qsim makes use of AVX/FMA vector operations, OpenMP multithreading, and
gate fusion [[1]](https://arxiv.org/abs/1601.07195)
[[2]](https://arxiv.org/abs/1704.01127)
to accelerate simulations. This performance is best demonstrated by the use
of qsim in cross-entropy benchmarks here:
[[3]](https://www.nature.com/articles/s41586-019-1666-5).

Integration with [Cirq](https://github.com/quantumlib/Cirq) makes getting 
started with qsim easy! Check out the
[install guide](https://github.com/quantumlib/qsim/blob/master/docs/install_qsimcirq.md)
or try the runnable
[notebook tutorial](https://github.com/quantumlib/qsim/blob/master/docs/tutorials/qsimcirq.ipynb).

## Design

This repository includes two top-level libraries for simulation:

-   **qsim** is a Schrödinger state-vector simulator designed to run on a
    single machine. It produces the full state vector as output which, 
    for instance, allows users to sample repeatedly from a single execution.
-   **qsimh** is a hybrid Schrödinger-Feynman simulator built for parallel
    execution on a cluster of machines. It produces amplitudes for user-
    specified output bitstrings.

These libraries can be invoked either directly or through the qsim-Cirq 
interface to perform the following operations:

-   Determine the final state vector of a circuit (qsim only).
-   Sample results from a circuit. Multiple samples can be generated with
    minimal additional cost for circuits with no intermediate measurements
    (qsim only).
-   Calculate amplitudes for user-specified result bitstrings. With qsimh,
    this is trivially parallelizable across several machines.

Circuits of up to 30 qubits can be simulated in qsim with ~16GB of RAM;
each additional qubit doubles the RAM requirement. In contrast, careful
use of qsimh can support 50 qubits or more.


[[1]](https://arxiv.org/abs/1601.07195) M. Smelyanskiy, N. P. Sawaya,
A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing
Environment", arXiv:1601.07195 (2016).

[[2]](https://arxiv.org/abs/1704.01127) T. Häner, D. S. Steiger,
"0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit", arXiv:1704.01127
(2017).

[[3]](https://www.nature.com/articles/s41586-019-1666-5), F. Arute et al,
"Quantum Supremacy Using a Programmable Superconducting Processor",
Nature 574, 505, (2019).
