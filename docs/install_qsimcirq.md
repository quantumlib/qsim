# Installing qsimcirq

The qsim-Cirq Python interface is available as a PyPI package for Linux, MacOS and Windows users.
For all others, Dockerfiles are provided to install qsim in a containerized
environment.

**Note:** The core qsim library (under
[lib/](https://github.com/quantumlib/qsim/blob/master/lib)) can be included
directly in C++ code without building and installing the qsimcirq interface.

## Before installation

Prior to installation, consider opening a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Prerequisites are included in the
[`requirements.txt`](https://github.com/quantumlib/qsim/blob/master/requirements.txt)
file, and will be automatically installed along with qsimcirq.

If you'd like to develop qsimcirq, a separate set of dependencies are includes
in the
[`dev-requirements.txt`](https://github.com/quantumlib/qsim/blob/master/dev-requirements.txt)
file. You can install them with `pip3 install -r dev-requirements.txt` or
`pip3 install qsimcirq[dev]`.

## Linux installation

We provide `qsimcirq` Python wheels on 64-bit `x86` architectures with
`Python 3.{10,11,12,13}`. The installation process will automatically check for
CUDA and GPUs on your computer if they exist and attempt to build a version of
qsim that can make use of the GPU(s). (Note that this is presently an
installation-time action and will take several minutes to finish.)

Simply run `pip3 install qsimcirq`.

## MacOS installation

We provide `qsimcirq` Python wheels on `x86` and Apple Silicon architectures
with `Python 3.{10,11,12,13}`.

Simply run `pip3 install qsimcirq`.

Note that, due to architectural differences, CUDA support is not available on
MacOS. The version of `qsimcirq` on MacOS will only use the CPU, without GPU
acceleration.

## Windows installation

We provide `qsimcirq` Python wheels on 64-bit `x86` and `amd64` architectures
with `Python 3.{10,11,12,13}`.

Simply run `pip3 install qsimcirq`.

## Conda Installation

`qsimcirq` is also available on conda-forge for Linux x86 including CUDA builds
and MacOS x86 and Apple Silicon ARM64. To install `qsimcirq` using conda, you
can use the following command:

```
conda install -c conda-forge qsimcirq
```

This will install the `qsimcirq` package from the conda-forge channel.

## Help! There's no compatible wheel for my machine!

If existing wheels do no meet your needs, please open an issue with your
machine configuration (i.e., CPU architecture, Python version) and consider
using the [Docker config](./docker.md) provided in the qsim GitHub repository.

## Testing

After installing `qsimcirq` on your machine, you can test the installation by
copying [qsimcirq_tests/qsimcirq_test.py](qsimcirq_tests/qsimcirq_test.py)
to your machine and running `python3 -m pytest qsimcirq_test.py`.

The file `qsimcirq_test.py` also has examples of how to use qsimcirq.

**Note:** Because of how Python searches for modules, the test file cannot
be run from inside a clone of the qsim repository, or from any parent
directory of such a repository. Failure to meet this criteria may result
in misbehaving tests (e.g., false positives after a failed installation).
