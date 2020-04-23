TARGETS = qsim
TESTS = run-cxx-tests

CXX=g++
CXXFLAGS = -O3 -march=native -fopenmp
PYBIND11 = true

export CXX
export CXXFLAGS

ifeq ($(PYBIND11), true)
  TARGETS += pybind
  TESTS += run-py-tests
endif

.PHONY: all
all: $(TARGETS)

.PHONY: qsim
qsim:
	$(MAKE) -C apps/

.PHONY: pybind
pybind:
	$(MAKE) -C pybind_interface/ pybind

.PHONY: cxx-tests
tests:
	-git submodule update --init --recursive tests/googletest
	$(MAKE) -C tests/

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	$(MAKE) -C tests/ run-all

PYTESTS = $(shell find qsimcirq_tests/ -name '*_test.py')

.PHONY: run-py-tests
run-py-tests: pybind
	for exe in $(PYTESTS); do if ! python3 -m pytest $$exe; then exit 1; fi; done

.PHONY: run-tests
run-tests: $(TESTS)

.PHONY: clean
clean:
	-$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C pybind_interface/ clean
