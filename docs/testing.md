# Testing qsim

Unit and integration tests are provided for qsim and its Cirq interface. To run
all tests, simply run the command:
```
make run-tests
```

**NOTE:** This command (and all others specified on this page) runs all tests in
sequence. If any test fails, execution will halt and no further tests will run.

## C++ tests

If you only want to run tests for the core C++ libraries, use this command:
```
make run-cxx-tests
```

To build tests without running them, instead use:
```
make cxx-tests
```

## Cirq interface tests

Similarly, tests specific to the Python Cirq interface can be run with:
```
make run-py-tests
```

**NOTE:** Due to how Python handles imports, this will fail if run from any
directory except the top-level `qsim` directory. Similarly, attempting to run
tests using an installed version of qsimcirq (e.g. with `pip3 install qsimcirq`)
can misbehave if there is a `qsimcirq/` directory _anywhere_ in the current
working directory. For more information, see the
[Python module documentation](https://docs.python.org/3/tutorial/modules.html#the-module-search-path).
