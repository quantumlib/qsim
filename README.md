<div align="center">

# qsim

High-performance quantum circuit simulator for C++ and Python.

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/qsim/blob/main/LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B17-fcbc2c.svg?logo=c%2B%2B&logoColor=white&style=flat-square&label=C%2B%2B)
[![qsimcirq project on
PyPI](https://img.shields.io/pypi/v/qsimcirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=e57430)](https://pypi.org/project/qsimcirq)
[![Compatible with Python versions 3.10 and
higher](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4023103-gray.svg?label=DOI&logo=doi&logoColor=white&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4023103)

[Features](#features) &ndash;
[Usage](#usage) &ndash;
[Documentation](#documentation) &ndash;
[Citing qsim](#how-to-cite-qsim) &ndash;
[Contact](#contact)

</div>

_qsim_ is a state-vector simulator for quantum circuits. Also known as a
_Schrödinger_ simulator, it represents a quantum state as a vector of
complex-valued amplitudes (a _state vector_) and applies matrix-vector
multiplication to simulate transformations that evolve the state over time.

qsim was used to produce landmark cross-entropy benchmark results published
in 2019 (Arute et al., "Quantum Supremacy Using a Programmable Superconducting
Processor", [Nature
vol.&nbsp;574](https://www.nature.com/articles/s41586-019-1666-5), 2019).

## Features

Being a _full_ state-vector simulator means that qsim computes all the
_2<sup> n</sup>_ amplitudes of the state vector, where _n_ is the number of
qubits. The total runtime is proportional to _g2<sup> n</sup>_, where _g_ is
the number of 2-qubit gates.

*   To speed up simulation, qsim uses gate fusion (Smelyanskiy et al.,
    [arXiv:1601.07195](https://arxiv.org/abs/1601.07195), 2016; Häner and
    Steiger, [arXiv:1704.01127](https://arxiv.org/abs/1704.01127), 2017) and
    single-precision arithmetic.

*   qsim is highly tuned to take advantage of CPU vector instruction sets (SSE,
    AVX/FMA, AVX-512, and ARM NEON) and OpenMP multithreading on computers that
    provide them, as well as GPUs when available (NVIDIA CUDA, NVIDIA
    cuStateVec from the cuQuantum SDK, and AMD HIP).

*   In addition to noiseless simulation, qsim can simulate noisy circuits using
    quantum trajectories with user-defined channels, and it includes a Matrix
    Product State (MPS) simulator for approximate simulation of larger
    circuits.

*   qsim includes a [Cirq](https://quantumai.google/cirq) interface (`qsimcirq`)
    and can be used to simulate quantum circuits written in Cirq.

## Usage

### C++ usage

The core qsim library (located in the source repository under the
[`lib/`](https://github.com/quantumlib/qsim/tree/main/lib) subdirectory) is
designed to be included directly in users' C++ applications, without installing
the Python interface. The sample applications provided in the qsim
[apps](https://github.com/quantumlib/qsim/tree/main/apps) directory can be used
as starting points. More information about the sample applications can be found
in the qsim
[documentation](https://github.com/quantumlib/qsim/blob/main/docs/usage.md).

### Python usage with Cirq

[Cirq](https://github.com/quantumlib/Cirq) is a framework for modeling and
invoking Noisy Intermediate-Scale Quantum (NISQ) circuits. Cirq can use qsim
as its simulation library through the qsim-Cirq Python interface, which is
called `qsimcirq` and is available as a PyPI package for Linux, macOS, and
Windows users. It can be installed by using the following command:

```shell
pip install qsimcirq
```

To get started with simulating Cirq circuits using qsim, please refer to the
[tutorial](https://github.com/quantumlib/qsim/blob/main/docs/tutorials/qsimcirq.ipynb).
More detailed information about the qsim-Cirq API can be found in the
[docs](https://github.com/quantumlib/qsim/blob/main/docs/cirq_interface.md).

### Unit tests

Unit tests for C++ libraries use the
[GoogleTest](https://github.com/google/googletest) framework, and are located in
[tests](https://github.com/quantumlib/qsim/tree/main/tests). Python tests use
[pytest](https://docs.pytest.org/en/stable/), and are located in
[qsimcirq_tests](https://github.com/quantumlib/qsim/tree/main/qsimcirq_tests).

Building and running the tests requires a C++17 compiler, `make`, `cmake` (to
build GoogleTest), `wget` or `curl` (to download Eigen), and the Python
development dependencies, which include `pybind11` and `pytest`. From the
repository root, the Python dependencies can be installed by running:

```shell
pip install -r requirements.txt --group dev
```

To build and run all tests, run the following command from the repository root:

```shell
make run-tests
```

This will compile all C++ test binaries to files with `.x` extensions and run
them one at a time, stopping early if a test fails. It will then run the tests
of the `qsimcirq` Python interface using `pytest`, which runs them in parallel.
To run C++ or Python tests only, run `make run-cxx-tests` or
`make run-py-tests`, respectively.

To clean up generated test files, run `make clean` from the repository root.

The tests can also be built and run using [Bazel](https://bazel.build) with the
command `bazel test tests:all`; see the [Bazel
documentation](https://github.com/quantumlib/qsim/blob/main/docs/bazel.md) for
details.

## Documentation

Please visit the [qsim documentation site](https://quantumai.google/qsim) for
guides, tutorials, and API reference documentation.

## How to cite qsim

When publishing articles or otherwise writing about qsim, please cite the qsim
version you use – it will help others reproduce your results. We use Zenodo to
preserve releases. The following links let you download the bibliographic
record for the latest stable release of qsim in some popular formats:

<div align="center">

[![Download BibTeX bibliography record for latest qsim
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://zenodo.org/records/22217864/export/bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest qsim
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://zenodo.org/records/22217864/export/csl)

</div>

For formatted citations and records in other formats, as well as records for all
releases of qsim past and present, please visit the [qsim page on
Zenodo](https://doi.org/10.5281/zenodo.4023103).

If you would like to cite a publication about qsim, please use Isakov et al.,
"Simulations of Quantum Circuits with Approximate Noise using qsim and Cirq",
[arXiv:2111.02396](https://arxiv.org/abs/2111.02396), 2021. This is also the
preferred citation listed in the repository's
[`CITATION.cff`](https://github.com/quantumlib/qsim/blob/main/CITATION.cff)
file.

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
         src="https://raw.githubusercontent.com/quantumlib/qsim/refs/heads/main/docs/images/quantum-ai-vertical.svg">
  </a>
</div>
