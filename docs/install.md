# Installing qsim

The qsim-Cirq interface is available as a PyPI package for Linux users. For all
other users, Dockerfiles are provided to install qsim in a contained
environment.

## Linux installation

Prior to installation, consider opening a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

qsim uses [CMake](https://cmake.org/) to ensure stable compilation of its C++
libraries across a variety of Linux distributions. CMake can be installed from
their website, or with the command `apt-get install cmake`.

TODO(@95-martin-orion): update to include pybind11 requirements.

To install qsim on Linux, simply run `pip3 install qsimcirq`. For examples of
how to use this package, see the tests in
[qsim/interfaces/tests/](interfaces/tests/).

## MacOS and Windows installation

qsim testing is currently limited to Linux. For users interested in running
qsim on a MacOS or Windows device, we recommend user the
[Docker config](docs/docker.md) provided with this repository.
