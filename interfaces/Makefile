# The name of the shared library that results after compiling qsim for Pybind11
QSIMLIB = qsimcirq/qsim`python3-config --extension-suffix`

PYBINDDIR = pybind_interface

# The flags for the compilation of the Pybind11 interface
PYBINDFLAGS = -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`

.PHONY: pybind
pybind:
	$(CXX) $(PYBINDDIR)/pybind_main.cpp -o $(QSIMLIB) $(CXXFLAGS) $(PYBINDFLAGS)

TESTS = $(shell find tests/ -name '*_test.py')

# WARNING: Python imports depend on your local directory! To ensure correct
# behavior, only run these tests from the top-level qsim Makefile.
.PHONY: run-tests
run-tests: pybind
	for exe in $(TESTS); do if ! python3 -m pytest $$exe; then exit 1; fi; done

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod $(QSIMLIB)