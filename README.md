<div align="center">

# qsim

High-performance quantum circuit simulator for C++ and Python.

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

_qsim_ is a state-vector simulator for quantum circuits. It is highly tuned to
take advantage of vector arithmetic instruction sets and multithreading on
computers that provide them, as well as GPUs when available. qsim also provides
a [Cirq](https://quantumai.google/cirq) interface (`qsimcirq`) and can be used
to simulate quantum circuits written in Cirq.

## Introduction

qsim is a Schrödinger full state-vector simulator: it models quantum
computations by representing the quantum state of a system as a vector of
complex numbers (a _state vector_) and evolving it through the application of
quantum gates. One matrix-vector multiplication corresponds to the application
of one gate. Essentially, the simulator performs matrix-vector multiplications
repeatedly.

Being a _full_ state-vector simulator means that qsim computes all the
*2<sup> n</sup>* amplitudes of the state vector, where *n* is the number of
qubits. The total runtime is proportional to *g2<sup> n</sup>*, where *g* is the
number of 2-qubit gates. To speed up simulation, qsim uses gate fusion
(Smelyanskiy et al., [arXiv:1601.07195](https://arxiv.org/abs/1601.07195), 2016;
Häner and Steiger, [arXiv:1704.01127](https://arxiv.org/abs/1704.01127), 2017),
single-precision arithmetic, AVX/FMA instructions for vectorization, and
OpenMP for multithreading (on hardware that provides those features).

qsim was used to produce landmark cross-entropy benchmarking results published
in 2019 (Arute et al., "Quantum Supremacy Using a Programmable Superconducting
Processor", [Nature
vol.&nbsp;574](https://www.nature.com/articles/s41586-019-1666-5), 2019).

## Usage

### C++ usage

The code is basically designed as a library. The user can modify sample
applications in [apps](https://github.com/quantumlib/qsim/tree/master/apps)
to meet their own needs. The usage of sample applications is described in the
[docs](https://github.com/quantumlib/qsim/blob/master/docs/usage.md).

### Python usage

The qsim-Cirq Python interface is called `qsimcirq` and is available as a PyPI
package for Linux, MacOS and Windows users. It can be installed by using the
following command:

```shell
pip install qsimcirq
```

`qsimcirq` is also available for Conda for Linux and MacOS. To install it from
conda-forge, you can use the following command:

```shell
conda install -c conda-forge qsimcirq
```

_Note_: The core qsim library (located in the source repository under the
[`lib/`](https://github.com/quantumlib/qsim/blob/master/lib) subdirectory) can
be included directly in C++ programs without installing the Python interface.

### Cirq usage

[Cirq](https://github.com/quantumlib/cirq) is a framework for modeling and
invoking Noisy Intermediate-Scale Quantum (NISQ) circuits. Cirq can use qsim
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
