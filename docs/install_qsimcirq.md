# Installing qsimcirq

The qsim-Cirq Python interface is available as a PyPI package for Linux users.
For all other users, Dockerfiles are provided to install qsim in a contained
environment.

**Note:** The core qsim library (under [lib/](/lib)) can be included directly
in C++ code without installing this interface.

## Linux installation

Prior to installation, consider opening a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

The qsim-Cirq interface uses [CMake](https://cmake.org/) to ensure stable
compilation of its C++ libraries across a variety of Linux distributions.
CMake can be installed from their website, or with the command
`apt-get install cmake`.

Other prerequisites (including pybind11 and pytest) are included in the
[`requirements.txt`](/requirements.txt) file, and will be automatically
installed along with qsimcirq.

To install the qsim-Cirq interface on Linux, simply run `pip3 install qsimcirq`.
For examples of how to use this package, see the tests in
[qsim/qsimcirq_tests/](/qsimcirq_tests/).

## MacOS and Windows installation

For users interested in running qsim on a MacOS or Windows device, we strongly
recommend using the [Docker config](/docs/docker.md) provided with this
repository.

### Experimental install process

Alternatively, MacOS and Windows users can follow the Linux install process,
but it is currently untested on those platforms. Users are encouraged to report
any issues seen with this process.

## Testing

After installing qsimcirq on your machine, you can test the installation by
copying [qsimcirq_tests/qsimcirq_test.py](qsimcirq_tests/qsimcirq_test.py)
to your machine and running `python3 -m pytest qsimcirq_test.py`.

**Note:** Because of how Python searches for modules, the test file cannot
be run from inside a clone of the qsim repository, or from any parent
directory of such a repository. Failure to meet this criteria may result
in misbehaving tests (e.g. false positives after a failed install).
