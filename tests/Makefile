HAVE_AVX512 = $(shell grep avx512f /proc/cpuinfo)
CXX_TARGETS = $(shell\
  if [ -z "$(HAVE_AVX512)" ] ; then\
    find . -maxdepth 1 -name "*_test.cc" ! -name "*512*";\
  else\
    find . -maxdepth 1 -name "*_test.cc";\
  fi\
)
CXX_TARGETS := $(CXX_TARGETS:%.cc=%.x)

CUDA_TARGETS = $(shell find . -maxdepth 1 -name "*cuda_test.cu")
CUDA_TARGETS := $(CUDA_TARGETS:%cuda_test.cu=%cuda_test.x)

CUSTATEVEC_TARGETS = $(shell find . -maxdepth 1 -name "*custatevec_test.cu")
CUSTATEVEC_TARGETS := $(CUSTATEVEC_TARGETS:%custatevec_test.cu=%custatevec_test.x)

GTEST_DIR = $(CURDIR)/googletest/googletest
GMOCK_DIR = $(CURDIR)/googletest/googlemock

CMAKE=cmake

TESTFLAGS = -I$(GTEST_DIR)/include -L$(GTEST_DIR)/make/lib -lgtest

.PHONY: cxx-tests
cxx-tests: $(CXX_TARGETS)

.PHONY: cuda-tests
cuda-tests: $(CUDA_TARGETS)

.PHONY: custatevec-tests
custatevec-tests: $(CUSTATEVEC_TARGETS)

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	for exe in $(CXX_TARGETS); do if ! ./$$exe; then exit 1; fi; done

.PHONY: run-cuda-tests
run-cuda-tests: cuda-tests
	for exe in $(CUDA_TARGETS); do if ! ./$$exe; then exit 1; fi; done

.PHONY: run-custatevec-tests
run-custatevec-tests: custatevec-tests
	for exe in $(CUSTATEVEC_TARGETS); do if ! ./$$exe; then exit 1; fi; done

$(GTEST_DIR)/make:
	-git submodule update --init --recursive googletest
	mkdir -p $(GTEST_DIR)/make
	cd $(GTEST_DIR)/make &&	$(CMAKE) .. && $(MAKE)

%.x: %.cc $(GTEST_DIR)/make
	$(CXX) -o ./$@ $< $(TESTFLAGS) $(CXXFLAGS) $(ARCHFLAGS)

%cuda_test.x: %cuda_test.cu $(GTEST_DIR)/make
	$(NVCC) -o ./$@ $< $(TESTFLAGS) $(NVCCFLAGS)

%custatevec_test.x: %custatevec_test.cu $(GTEST_DIR)/make
	$(NVCC) -o ./$@ $< $(TESTFLAGS) $(NVCCFLAGS) $(CUSTATEVECFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
	rm -rf $(GTEST_DIR)/make
