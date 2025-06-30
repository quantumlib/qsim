# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Version info for the copy of Eigen we will download and build locally.
EIGEN_PREFIX = "3bb6a48d8c171cf20b5f8e48bfb4e424fbd4f79e"
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/"

# Default build targets. Additional may be added conditionally below.
TARGETS = qsim
TESTS = run-cxx-tests

# By default, we also build the pybind11-based Python interface.
PYBIND11 ?= true

ifeq ($(PYBIND11), true)
    TARGETS += pybind
    TESTS += run-py-tests
endif

# Default options for Pytest (only used if the pybind interface is built).
PYTESTFLAGS ?= -v

# Default C++ compilers and compiler flags. Can be overriden via env variables.
CXX ?= g++
NVCC ?= nvcc
HIPCC ?= hipcc

CXXFLAGS ?= -O3 -std=c++17 -fopenmp -flto=auto
NVCCFLAGS ?= -O3 --std c++17 -Wno-deprecated-gpu-targets
HIPCCFLAGS ?= -O3

# For compatibility with CMake, if $CUDAARCHS is set, use it to set the
# architecture options to nvcc. Otherwise, default to the "native" option,
# which is what our pybind11_interface/GetCUDAARCHS.cmake also does.
ifneq (,$(CUDAARCHS))
    # Avoid a common mistake that leads to difficult-to-diagnose errors.
    COMMA := ,
    ifeq ($(COMMA),$(findstring $(COMMA),$(CUDAARCHS)))
        $(error Error: the value of the CUDAARCHS environment variable \
                must use semicolons as separators, not commas)
    endif
    ARCHS := $(subst ;, ,$(CUDAARCHS))
    NVCCFLAGS += $(foreach a,$(ARCHS),--generate-code arch=compute_$(a),code=sm_$(a))
else
    NVCCFLAGS += -arch=native
endif

# Determine whether we can include CUDA and cuStateVec support. We build for
# CUDA if (i) we find $NVCC or (ii) $CUDA_PATH is set. For cuStateVec, there's
# no way to find the cuQuantum libraries other than by being told, so we rely
# on the user or calling environment to set variable $CUQUANTUM_ROOT.

ifneq (,$(shell which $(NVCC)))
    # nvcc adds appropriate -I and -L flags, so nothing more is needed here.
    TARGETS += qsim-cuda
    TESTS += run-cuda-tests
else
    ifneq (,$(strip $(CUDA_PATH)))
        # $CUDA_PATH is set. Check that the path truly does exist.
        ifneq (,$(strip $(wildcard $(CUDA_PATH)/.)))
            # $CUDA_PATH is set, but we know we didn't find nvcc on the user's
            # $PATH or as an absolute path (if $NVCC was set to a full path).
	    # Try the safest choice for finding nvcc & give up if that fails.
            NVCC = $(CUDA_PATH)/bin/nvcc
            ifneq (,$(strip $(wildcard $(NVCC))))
                CXXFLAGS += -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64
                TARGETS += qsim-cuda
                TESTS += run-cuda-tests
            else
                $(warning nvcc not found, so cannot build CUDA interfaces)
            endif
        else
            $(warning $$CUDA_PATH is set, but the path does not seem to exist)
        endif
    endif
endif

ifneq (,$(strip $(CUQUANTUM_ROOT)))
    # $CUQUANTUM_ROOT is set. Check that the path truly does exist.
    ifneq (,$(strip $(wildcard $(CUQUANTUM_ROOT)/.)))
        CUSVFLAGS =  -I$(CUQUANTUM_ROOT)/include
        CUSVFLAGS += -L${CUQUANTUM_ROOT}/lib -L$(CUQUANTUM_ROOT)/lib64
        CUSVFLAGS += -lcustatevec -lcublas
        CUSTATEVECFLAGS ?= $(CUSVFLAGS)
        TARGETS += qsim-custatevec
        TESTS += run-custatevec-tests
    else
        $(warning $$CUQUANTUM_ROOT is set, but the path does not seem to exist)
    endif
endif

# Export all variables to subprocesses without having to export them individually.
.EXPORT_ALL_VARIABLES:

# The rest is build rules and make targets.

.PHONY: all
all: $(TARGETS)

.PHONY: qsim
qsim:
	$(MAKE) -C apps/ qsim

.PHONY: qsim-cuda
qsim-cuda:
	$(MAKE) -C apps/ qsim-cuda

.PHONY: qsim-custatevec
qsim-custatevec: | check-cuquantum-root-set
	$(MAKE) -C apps/ qsim-custatevec

.PHONY: qsim-hip
qsim-hip:
	$(MAKE) -C apps/ qsim-hip

.PHONY: pybind
pybind:
	$(MAKE) -C pybind_interface/ pybind

.PHONY: cxx-tests
cxx-tests: eigen
	$(MAKE) -C tests/ cxx-tests

.PHONY: cuda-tests
cuda-tests:
	$(MAKE) -C tests/ cuda-tests

.PHONY: custatevec-tests
custatevec-tests: | check-cuquantum-root-set
	$(MAKE) -C tests/ custatevec-tests

.PHONY: hip-tests
hip-tests:
	$(MAKE) -C tests/ hip-tests

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	$(MAKE) -C tests/ run-cxx-tests

.PHONY: run-cuda-tests
run-cuda-tests: cuda-tests
	$(MAKE) -C tests/ run-cuda-tests

.PHONY: run-custatevec-tests
run-custatevec-tests: custatevec-tests
	$(MAKE) -C tests/ run-custatevec-tests

.PHONY: run-hip-tests
run-hip-tests: hip-tests
	$(MAKE) -C tests/ run-hip-tests

PYTESTS := $(wildcard qsimcirq_tests/*_test.py)

.PHONY: run-py-tests
run-py-tests: pybind
	python3 -m pytest $(PYTESTFLAGS) $(PYTESTS)

.PHONY: run-tests tests
run-tests tests: $(TESTS)

.PHONY: check-cuquantum-root-set
check-cuquantum-root-set:
	@if [[ -z "$(CUQUANTUM_ROOT)" ]]; then \
	    echo Error: '$$CUQUANTUM_ROOT must be set in order to use cuStateVec.' \
	    exit 1 \
	fi

eigen:
	-rm -rf eigen
	wget $(EIGEN_URL)/$(EIGEN_PREFIX)/eigen-$(EIGEN_PREFIX).tar.gz
	tar -xzf eigen-$(EIGEN_PREFIX).tar.gz && mv eigen-$(EIGEN_PREFIX) eigen
	rm eigen-$(EIGEN_PREFIX).tar.gz

.PHONY: clean
clean:
	-rm -rf eigen
	-$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C pybind_interface/ clean

LOCAL_VARS = TARGETS TESTS PYTESTS PYTESTFLAGS CXX CXXFLAGS NVCC NVCCFLAGS $\
	HIPCC HIPCCFLAGS CUDA_PATH CUQUANTUM_ROOT CUSTATEVECFLAGS

.PHONY: print-vars
print-vars: ; @$(foreach n,$(sort $(LOCAL_VARS)),echo $n=$($n);)
