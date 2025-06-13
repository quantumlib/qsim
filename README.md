<div align="center">

# qsim and qsimh

High-performance quantum circuit simulators for C++ and Python.

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/qsim/blob/main/LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B17-fcbc2c.svg?logo=c%2B%2B&logoColor=white&style=flat-square&label=C%2B%2B)
[![qsim project on
PyPI](https://img.shields.io/pypi/v/qsim.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=e57430)](https://pypi.org/project/qsim)
[![Compatible with Python versions 3.10 and
higher](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4023103-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4023103)

[Features](#features) &ndash;
[Usage](#usage) &ndash;
[Documentation](#cirq-documentation) &ndash;
[Citing qsim](#citing-qsim) &ndash;
[Contact](#contact)

</div>

_qsim_ and _qsimh_ are Schrödinger and Schrödinger-Feynman state vector
simulators for quantum circuits. They are highly tuned to take advantage of
vector arithmetic instruction sets and multithreading on computers that provide
them, as well as GPUs when available. qsim also provides a
[Cirq](https://quantumai.google/cirq) interface (`qsimcirq`) and can be used to
simulate quantum circuits written in Cirq. These simulators were used to produce
landmark cross-entropy benchmarking results published in Nature
[[1]](https://www.nature.com/articles/s41586-019-1666-5).

[[1]](https://www.nature.com/articles/s41586-019-1666-5), F. Arute et al,
"Quantum Supremacy Using a Programmable Superconducting Processor",
Nature 574, 505, (2019).

## Features

qsim and qsimh (yes, they really are spelled in all lower case!) are two
closely-related libraries. They are described in more detail below.

### qsim

qsim is a Schrödinger full state-vector simulator. It computes all the *2<sup>n</sup>*
amplitudes of the state vector, where *n* is the number of qubits.
Essentially, the simulator performs matrix-vector multiplications repeatedly.
One matrix-vector multiplication corresponds to applying one gate.
The total runtime is proportional to *g2<sup>n</sup>*, where *g* is the number of
2-qubit gates. To speed up the simulator, we use gate fusion
[[2]](https://arxiv.org/abs/1601.07195) [[3]](https://arxiv.org/abs/1704.01127),
single precision arithmetic, AVX/FMA instructions for vectorization and OpenMP
for multithreading on hardware that provides those features.

[[2]](https://arxiv.org/abs/1601.07195) M. Smelyanskiy, N. P. Sawaya,
A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing
Environment", arXiv:1601.07195 (2016).

[[3]](https://arxiv.org/abs/1704.01127) T. Häner, D. S. Steiger,
"0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit", arXiv:1704.01127
(2017).

### qsimh

qsimh is a hybrid Schrödinger-Feynman simulator
[[4]](https://arxiv.org/abs/1807.10749). The lattice is split into two parts
and the Schmidt decomposition is used to decompose 2-qubit gates on the
cut. If the Schmidt rank of each gate is *m* and the number of gates on
the cut is *k* then there are *m<sup>k</sup>* paths. To simulate a circuit with
fidelity one, one needs to simulate all the *m<sup>k</sup>* paths and sum the results.
  The total runtime is proportional to *(2<sup>n<sub>1</sub></sup> + 2<sup>n<sub>2</sub></sup>)m<sup>k</sup>*, where *n<sub>1</sub>*
and *n<sub>2</sub>* are the qubit numbers in the first and second parts. Path
simulations are independent of each other and can be trivially parallelized
to run on supercomputers or in data centers. Note that one can run simulations
with fidelity *F < 1* just by summing over a fraction *F* of all the paths.

A two level checkpointing scheme is used to improve performance. Say, there
are *k* gates on the cut. We split those into three parts: *p+r+s=k*, where
*p* is the number of "prefix" gates, *r* is the number of "root" gates and
*s* is the number of "suffix" gates. The first checkpoint is executed after
applying all the gates up to and including the prefix gates and the second
checkpoint is executed after applying all the gates up to and including the
root gates. The full summation over all the paths for the root and suffix gates
is performed.

If *p>0* then one such simulation gives *F &#8776; m<sup>-p</sup>* (for all the
prefix gates having the same Schmidt rank *m*). One needs to run *m<sup>p</sup>*
simulations with different prefix paths and sum the results to get *F = 1*.

[[4]](https://arxiv.org/abs/1807.10749) I. L. Markov, A. Fatima, S. V. Isakov,
S. Boixo, "Quantum Supremacy Is Both Closer and Farther than It Appears",
arXiv:1807.10749 (2018).

## Usage

### C++ usage

The code is basically designed as a library. The user can modify sample
applications in [apps](https://github.com/quantumlib/qsim/tree/master/apps)
to meet their own needs. The usage of sample applications is described in the
[docs](https://github.com/quantumlib/qsim/blob/master/docs/usage.md).

### Python usage

The Python `qsimcirq` module provides a Python interface to the qsim library.

### Cirq usage

[Cirq](https://github.com/quantumlib/cirq) is a framework for modeling and
invoking Noisy Intermediate Scale Quantum (NISQ) circuits. Cirq can use qsim
as its simulation library. To get started with simulating Cirq circuits using
qsim, please refer to the
[tutorial](https://github.com/quantumlib/qsim/blob/master/docs/tutorials/qsimcirq.ipynb).

More detailed information about the qsim-Cirq API can be found in the
[docs](https://github.com/quantumlib/qsim/blob/master/docs/cirq_interface.md).

### Input format

> [!WARNING]
> This format is deprecated, and no longer actively maintained.

The circuit input format is described in the
[docs](https://github.com/quantumlib/qsim/blob/master/docs/input_format.md).

### Sample circuits

A number of sample circuits are provided in
[circuits](https://github.com/quantumlib/qsim/tree/master/circuits).

### Unit tests

Unit tests for C++ libraries use the
[GoogleTest](https://github.com/google/googletest) framework, and are located in
[tests](https://github.com/quantumlib/qsim/tree/master/tests). Python tests use
[pytest](https://docs.pytest.org/en/stable/), and are located in
[qsimcirq_tests](https://github.com/quantumlib/qsim/tree/master/qsimcirq_tests).

To build and run all tests, run:

```shell
make run-tests
```

This will compile all test binaries to files with `.x` extensions, and run each
test in series. Testing will stop early if a test fails. It will also run tests
of the `qsimcirq` python interface. To run C++ or python tests only, run
`make run-cxx-tests` or `make run-py-tests`, respectively.

To clean up generated test files, run `make clean` from the test directory

## qsim documentation

Please visit the [qsim documentation site](https://quantumai.google/qsim)
guides, tutorials, and API reference documentation.


## How to cite qsim<a name="how-to-cite-qsim"></a><a name="how-to-cite"></a>

Qsim is uploaded to Zenodo automatically. Click on this badge [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4023103.svg)](https://doi.org/10.5281/zenodo.4023103) to see all the citation formats for all versions.

An equivalent BibTeX format reference is below for all the versions:

```bibtex
@software{quantum_ai_team_and_collaborators_2020_4023103,
  author       = {Quantum AI team and collaborators},
  title        = {qsim},
  month        = Sep,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4023103},
  url          = {https://doi.org/10.5281/zenodo.4023103}
}
```

## Contact

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2019 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
