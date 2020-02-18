TARGETS = qsim
TESTS = run-cxx-tests

CXX=g++
CXXFLAGS = -O3 -mavx2 -mfma -fopenmp -lgomp
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
	$(MAKE) -C interfaces/ pybind

.PHONY: cxx-tests
tests:
	-git submodule update --init --recursive tests/googletest
	$(MAKE) -C tests/

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	$(MAKE) -C tests/ run-all

# TODO: support python tests
.PHONY: run-py-tests
run-py-tests: pybind
	$(MAKE) -C interfaces/ run-tests

.PHONY: run-tests
run-tests: $(TESTS)

.PHONY: clean
clean:
	-$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C interfaces/ clean