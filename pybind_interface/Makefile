# The name of the shared library that results after compiling qsim for Pybind11
QSIMLIB = ../qsimcirq/qsim`python3-config --extension-suffix`

# The flags for the compilation of the Pybind11 interface
PYBINDFLAGS = -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`

.PHONY: pybind
pybind:
	$(CXX) pybind_main.cpp -o $(QSIMLIB) $(CXXFLAGS) $(PYBINDFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod $(QSIMLIB)