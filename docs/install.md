# Installing qsim

The qsim-Cirq interface is available as a PyPI package for Linux users. For all
other users, Dockerfiles are provided to install qsim in a contained
environment.

## Linux installation

Prior to installation, consider opening a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

To install qsim on Linux, simply run `pip3 install qsimcirq`. For examples of
how to use this package, see the tests in
[qsim/interfaces/tests/](interfaces/tests/).

## MacOS and Windows installation

qsim is currently native to Linux. Users wishing to run qsim on a MacOS or
Windows device should rely on the [Docker tools](docs/docker.md) provided with
this repository.
