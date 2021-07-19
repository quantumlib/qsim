EIGEN_PREFIX = "d10b27fe37736d2944630ecd7557cefa95cf87c9"
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/"

TARGETS = qsim
TESTS = run-cxx-tests

CXX=g++
CXXFLAGS = -O3 -fopenmp
PYBIND11 = true

export CXX
export CXXFLAGS

ifeq ($(PYBIND11), true)
  TARGETS += pybind
  TESTS += run-py-tests
endif

.PHONY: download-eigen
download-eigen:
	$(shell\
		rm -rf eigen;\
		wget $(EIGEN_URL)/$(EIGEN_PREFIX)/eigen-$(EIGEN_PREFIX).tar.gz;\
		tar -xf eigen-$(EIGEN_PREFIX).tar.gz && mv eigen-$(EIGEN_PREFIX) eigen;\
		rm eigen-$(EIGEN_PREFIX).tar.gz;)

.PHONY: all
all: $(TARGETS)

.PHONY: qsim
qsim:
	$(MAKE) -C apps/

.PHONY: pybind
pybind:
	$(MAKE) -C pybind_interface/ pybind

.PHONY: cxx-tests
tests: download-eigen
	-git submodule update --init --recursive tests/googletest
	$(MAKE) -C tests/

.PHONY: run-cxx-tests
run-cxx-tests: download-eigen
	$(MAKE) -C tests/ run-all

PYTESTS = $(shell find qsimcirq_tests/ -name '*_test.py')

.PHONY: run-py-tests
run-py-tests: pybind
	for exe in $(PYTESTS); do if ! python3 -m pytest $$exe; then exit 1; fi; done

.PHONY: run-tests
run-tests: $(TESTS)

.PHONY: clean
clean:
	rm -rf eigen;
	-$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C pybind_interface/ clean
