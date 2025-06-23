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

SUFFIX := $(shell python3-config --extension-suffix)

# The names of the shared libraries that result after compiling qsim for Pybind11
QSIMLIB_BASIC = ../qsimcirq/qsim_basic$(SUFFIX)
QSIMLIB_SSE = ../qsimcirq/qsim_sse$(SUFFIX)
QSIMLIB_AVX2 = ../qsimcirq/qsim_avx2$(SUFFIX)
QSIMLIB_AVX512 = ../qsimcirq/qsim_avx512$(SUFFIX)
QSIMLIB_CUDA = ../qsimcirq/qsim_cuda$(SUFFIX)
QSIMLIB_CUSTATEVEC = ../qsimcirq/qsim_custatevec$(SUFFIX)
QSIMLIB_HIP = ../qsimcirq/qsim_hip$(SUFFIX)
QSIMLIB_DECIDE = ../qsimcirq/qsim_decide$(SUFFIX)

# Certain Pybind11 flags are common for all cases.
PYBIND_INCLUDES := $(shell pybind11-config --includes)
PYBINDFLAGS = -Wall -shared -std=c++17 -fPIC $(PYBIND_INCLUDES)

# GCC 12 produces false-positive array-bounds warnings in some code (see
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115824). Pybind 3.x automatically
# disables them (see https://github.com/pybind/pybind11/pull/5355/files). We're
# still using 2.x, so we get the warnings unless we disable them ourselves.
# TODO(mhucka): remove this once we can upgrade to Pybind11 version 3.x.
GCC_MAJOR_VERSION := $(firstword $(subst ., ,$(shell gcc -dumpversion)))
ifeq ($(GCC_MAJOR_VERSION), 12)
    PYBINDFLAGS += -Wno-array-bounds -Wno-stringop-overread
endif

# The flags for the compilation of the simd-specific Pybind11 interfaces
PYBINDFLAGS_BASIC = $(PYBINDFLAGS)
PYBINDFLAGS_SSE = -msse4.1 $(PYBINDFLAGS)
PYBINDFLAGS_AVX2 = -mavx2 -mfma $(PYBINDFLAGS)
PYBINDFLAGS_AVX512 = -mavx512f -mbmi2 $(PYBINDFLAGS)

# The flags for the compilation of CUDA-specific Pybind11 interfaces
PYBINDFLAGS_CUDA = -std=c++17 -x cu -Xcompiler "-Wall -shared -fPIC $(PYBIND_INCLUDES)"

# The flags for the compilation of cuStateVec-specific Pybind11 interfaces
PYBINDFLAGS_CUSTATEVEC = $(CUSTATEVECFLAGS) $(PYBINDFLAGS_CUDA)

# The flags for the compilation of HIP-specific Pybind11 interfaces
PYBINDFLAGS_HIP = -std=c++17 -Wall -shared -fPIC $(PYBIND_INCLUDES)

# Check for nvcc to decide compilation mode.
ifeq ($(shell which $(NVCC)),)
# Check for hipcc to decide compilation mode.
ifeq ($(shell which $(HIPCC)),)
pybind: pybind-cpu decide-cpu
else
pybind: pybind-hip decide-hip
endif
else
# Check for the cuStateVec library.
ifeq ($(CUQUANTUM_ROOT),)
pybind: pybind-cpu pybind-cuda decide-cuda
else
pybind: pybind-cpu pybind-cuda pybind-custatevec decide-custatevec
endif
endif

.PHONY: pybind-cpu
pybind-cpu:
	$(CXX) basic/pybind_main_basic.cpp -o $(QSIMLIB_BASIC) $(CXXFLAGS) $(PYBINDFLAGS_BASIC)
	$(CXX) sse/pybind_main_sse.cpp -o $(QSIMLIB_SSE) $(CXXFLAGS) $(PYBINDFLAGS_SSE)
	$(CXX) avx2/pybind_main_avx2.cpp -o $(QSIMLIB_AVX2) $(CXXFLAGS) $(PYBINDFLAGS_AVX2)
	$(CXX) avx512/pybind_main_avx512.cpp -o $(QSIMLIB_AVX512) $(CXXFLAGS) $(PYBINDFLAGS_AVX512)

.PHONY: decide-cpu
decide-cpu:
	echo "building decide-cpu"
	$(CXX) decide/decide.cpp -o $(QSIMLIB_DECIDE) $(CXXFLAGS) $(PYBINDFLAGS_BASIC)

.PHONY: pybind-cuda
pybind-cuda:
	$(NVCC) cuda/pybind_main_cuda.cpp -o $(QSIMLIB_CUDA) $(NVCCFLAGS) $(PYBINDFLAGS_CUDA)

.PHONY: decide-cuda
decide-cuda:
	$(NVCC) decide/decide.cpp -o $(QSIMLIB_DECIDE) $(NVCCFLAGS) $(PYBINDFLAGS_CUDA)

.PHONY: pybind-custatevec
pybind-custatevec:
	$(NVCC) custatevec/pybind_main_custatevec.cpp -o $(QSIMLIB_CUSTATEVEC) $(NVCCFLAGS) $(PYBINDFLAGS_CUSTATEVEC)

.PHONY: decide-custatevec
decide-custatevec:
	$(NVCC) decide/decide.cpp -D__CUSTATEVEC__ -o $(QSIMLIB_DECIDE) $(NVCCFLAGS) $(PYBINDFLAGS_CUDA)

.PHONY: pybind-hip
pybind-hip:
	$(HIPCC) basic/pybind_main_basic.cpp -o $(QSIMLIB_BASIC) $(CXXFLAGS) $(PYBINDFLAGS_BASIC)
	$(HIPCC) sse/pybind_main_sse.cpp -o $(QSIMLIB_SSE) $(CXXFLAGS) $(PYBINDFLAGS_SSE)
	$(HIPCC) avx2/pybind_main_avx2.cpp -o $(QSIMLIB_AVX2) $(CXXFLAGS) $(PYBINDFLAGS_AVX2)
	$(HIPCC) avx512/pybind_main_avx512.cpp -o $(QSIMLIB_AVX512) $(CXXFLAGS) $(PYBINDFLAGS_AVX512)
	$(HIPCC) hip/pybind_main_hip.cpp -o $(QSIMLIB_HIP) $(HIPCCFLAGS) $(PYBINDFLAGS_HIP)

.PHONY: decide-hip
decide-hip:
	$(HIPCC) decide/decide.cpp -o $(QSIMLIB_DECIDE) $(HIPCCFLAGS) $(PYBINDFLAGS_HIP)

.PHONY: clean
clean:
	-rm -f ./basic/*.x ./basic/*.a ./basic/*.so ./basic/*.mod $(QSIMLIB_BASIC)
	-rm -f ./sse/*.x ./sse/*.a ./sse/*.so ./sse/*.mod $(QSIMLIB_SSE)
	-rm -f ./avx2/*.x ./avx2/*.a ./avx2/*.so ./avx2/*.mod $(QSIMLIB_AVX2)
	-rm -f ./avx512/*.x ./avx512/*.a ./avx512/*.so ./avx512/*.mod $(QSIMLIB_AVX512)
	-rm -f ./cuda/*.x ./cuda/*.a ./cuda/*.so ./cuda/*.mod $(QSIMLIB_CUDA)
	-rm -f ./hip/*.x ./hip/*.a ./hip/*.so ./hip/*.mod $(QSIMLIB_HIP)
	-rm -f ./custatevec/*.x ./custatevec/*.a ./custatevec/*.so ./custatevec/*.mod $(QSIMLIB_CUSTATEVEC)
	-rm -f ./decide/*.x ./decide/*.a ./decide/*.so ./decide/*.mod $(QSIMLIB_DECIDE)
