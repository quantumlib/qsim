CXX_TARGETS = $(shell find . -maxdepth 1 -name '*.cc')
CXX_TARGETS := $(CXX_TARGETS:%.cc=%.x)

CUDA_TARGETS = $(shell find . -maxdepth 1 -name '*cuda.cu')
CUDA_TARGETS := $(CUDA_TARGETS:%cuda.cu=%cuda.x)

CUSTATEVEC_TARGETS = $(shell find . -maxdepth 1 -name "*custatevec.cu")
CUSTATEVEC_TARGETS := $(CUSTATEVEC_TARGETS:%custatevec.cu=%custatevec.x)

CUSTATEVECEX_TARGETS = $(shell find . -maxdepth 1 -name "*custatevecex.cu")
CUSTATEVECEX_TARGETS := $(CUSTATEVECEX_TARGETS:%custatevecex.cu=%custatevecex.x)

HIP_TARGETS = $(shell find . -maxdepth 1 -name '*cuda.cu')
HIP_TARGETS := $(HIP_TARGETS:%cuda.cu=%hip.x)

# Keep qsim_base's build unchanged, but enable the backend's cooperative
# inner-thread mode for the separate tiled executable when the compiler has
# OpenMP support.
TILED_OPENMP_FLAGS := $(shell $(CXX) -fopenmp -x c++ -E /dev/null -o /dev/null 2>/dev/null && echo -fopenmp)
TILED_ARCH_FLAGS := $(shell $(CXX) -march=native -x c++ -E /dev/null -o /dev/null 2>/dev/null && echo -march=native)

.PHONY: qsim
qsim: $(CXX_TARGETS)

.PHONY: tiled
tiled: qsim_tiled.x

qsim_tiled.x: override CXXFLAGS += $(TILED_OPENMP_FLAGS) $(TILED_ARCH_FLAGS)

.PHONY: benchmark
benchmark: qsim_tiled.x qsim_base.x
	bash ./benchmark_tiled.sh

.PHONY: qsim-cuda
qsim-cuda: $(CUDA_TARGETS)

.PHONY: qsim-custatevec
qsim-custatevec: $(CUSTATEVEC_TARGETS)

.PHONY: qsim-custatevecex
qsim-custatevecex: $(CUSTATEVECEX_TARGETS)

.PHONY: qsim-hip
qsim-hip: $(HIP_TARGETS)

%.x: %.cc
	$(CXX) -o ./$@ $< $(CXXFLAGS) $(ARCHFLAGS)

%cuda.x: %cuda.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS)

%custatevec.x: %custatevec.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS) $(CUSTATEVECFLAGS)

%custatevecex.x: %custatevecex.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS) $(CUSTATEVECFLAGS)

%hip.x: %cuda.cu
	$(HIPCC) -o ./$@ $< $(HIPCCFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
