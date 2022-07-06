# Installing qsimcirq

The qsim-Cirq Python interface is available as a PyPI package for Linux, MacOS and Windows users.
For all others, Dockerfiles are provided to install qsim in a contained
environment.

**Note:** The core qsim library (under
[lib/](https://github.com/quantumlib/qsim/blob/master/lib)) can be included
directly in C++ code without installing this interface.

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

We provide `qsimcirq` Python wheels on 64-bit `x86` architectures with `Python 3.{6,7,8,9}`.

Simply run `pip3 install qsimcirq`.

## MacOS installation

We provide `qsimcirq` Python wheels on `x86` architectures with `Python 3.{6,7,8,9}`.

Simply run `pip3 install qsimcirq`.

## Windows installation

We provide `qsimcirq` Python wheels on 64-bit `x86` and `amd64` architectures with `Python 3.{6,7,8,9}`.

Simply run `pip3 install qsimcirq`.

## There's no compatible wheel for my machine!

If existing wheels do no meet your needs please open an issue with your machine configuration (i.e. CPU architecture, Python version) and consider using the [Docker config](./docker.md) provided with this repository.

## Testing

After installing `qsimcirq` on your machine, you can test the installation by
copying [qsimcirq_tests/qsimcirq_test.py](qsimcirq_tests/qsimcirq_test.py)
to your machine and running `python3 -m pytest qsimcirq_test.py`.

It also has examples of how how to use this package.

**Note:** Because of how Python searches for modules, the test file cannot
be run from inside a clone of the qsim repository, or from any parent
directory of such a repository. Failure to meet this criteria may result
in misbehaving tests (e.g. false positives after a failed install).
