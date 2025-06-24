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

# Determine the hardware features available in this CPU.
HAVE_SSE := $(shell grep -q sse /proc/cpuinfo && echo "true")
HAVE_AVX2 := $(shell grep -q avx2 /proc/cpuinfo && echo "true")
HAVE_AVX512 := $(shell grep -q avx512f /proc/cpuinfo && echo "true")

# Default targets. Always built.
BASIC_FILES := $(shell ls *.cc | egrep -v '_avx|_sse')

# Additional flags and targets for non-CUDA cases.
SSE_FILES =
AVX2_FILES =
AVX512_FILES =
ifneq (,$(HAVE_SSE))
    SSE_FLAGS ?= -msse4.1
    SSE_FILES := $(wildcard *_sse_test.cc)
endif
ifneq (,$(HAVE_AVX2))
    AVX2_FLAGS ?= -mavx2 -mfma
    AVX2_FILES := $(wildcard *_avx_test.cc)
endif
ifneq (,$(HAVE_AVX512))
    AVX512_FLAGS ?= -mavx512f -mbmi2
    AVX512_FILES := $(wildcard *_avx512_test.cc)
endif

CXX_FILES := $(BASIC_FILES) $(SSE_FILES) $(AVX2_FILES) $(AVX512_FILES)
CXX_TARGETS := $(CXX_FILES:%.cc=%.x)
CXXFLAGS := $(CXXFLAGS) $(SSE_FLAGS) $(AVX2_FLAGS) $(AVX512_FLAGS)

CUDA_FILES := $(wildcard *cuda_test.cu)
CUDA_TARGETS := $(CUDA_FILES:%cuda_test.cu=%cuda_test.x)

CUSTATEVEC_FILES := $(wildcard *custatevec_test.cu)
CUSTATEVEC_TARGETS := $(CUSTATEVEC_FILES:%custatevec_test.cu=%custatevec_test.x)

HIP_FILES := $(wildcard *cuda_test.cu)
HIP_TARGETS := $(HIP_FILES:%cuda_test.cu=%hip_test.x)

GTEST_DIR := $(CURDIR)/googletest/googletest
GMOCK_DIR := $(CURDIR)/googletest/googlemock

TESTFLAGS = -I$(GTEST_DIR)/include -L$(GTEST_DIR)/build/lib -lgtest

CMAKE = cmake

.PHONY: cxx-tests
cxx-tests: $(CXX_TARGETS)

.PHONY: cuda-tests
cuda-tests: $(CUDA_TARGETS)

.PHONY: custatevec-tests
custatevec-tests: $(CUSTATEVEC_TARGETS)

.PHONY: hip-tests
hip-tests: $(HIP_TARGETS)

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	for exe in $(CXX_TARGETS); do if ! ./$$exe; then exit 1; fi; done

.PHONY: run-cuda-tests
run-cuda-tests: cuda-tests
	for exe in $(CUDA_TARGETS); do if ! ./$$exe; then exit 1; fi; done

.PHONY: run-custatevec-tests
run-custatevec-tests: custatevec-tests
	for exe in $(CUSTATEVEC_TARGETS); do if ! ./$$exe; then exit 1; fi; done

.PHONY: run-hip-tests
run-hip-tests: hip-tests
	for exe in $(HIP_TARGETS); do if ! ./$$exe; then exit 1; fi; done

$(GTEST_DIR)/build:
	-git submodule update --init --recursive googletest
	mkdir -p $(GTEST_DIR)/build
	cd $(GTEST_DIR) && $(CMAKE) -B build -S ..
	cd $(GTEST_DIR)/build && make

%.x: %.cc $(GTEST_DIR)/build
	$(CXX) -o ./$@ $< $(TESTFLAGS) $(CXXFLAGS) $(ARCHFLAGS)

%cuda_test.x: %cuda_test.cu $(GTEST_DIR)/build
	$(NVCC) -o ./$@ $< $(TESTFLAGS) $(NVCCFLAGS)

%custatevec_test.x: %custatevec_test.cu $(GTEST_DIR)/build
	$(NVCC) -o ./$@ $< $(TESTFLAGS) $(NVCCFLAGS) $(CUSTATEVECFLAGS)

%hip_test.x: %cuda_test.cu $(GTEST_DIR)/build
	$(HIPCC) -o ./$@ $< $(TESTFLAGS) $(HIPCCFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
	rm -rf $(GTEST_DIR)/build

LOCAL_VARS = HAVE_SSE HAVE_AVX2 HAVE_AVX512 SSE_FLAGS AVX2_FLAGS $\
	AVX512_FLAGS CXXFLAGS CXX_TARGETS TEST_FLAGS

.PHONY: print-vars
print-vars: ; @$(foreach n,$(sort $(LOCAL_VARS)),echo $n=$($n);)
