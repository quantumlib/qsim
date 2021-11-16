CXX_TARGETS = $(shell find . -maxdepth 1 -name '*.cc')
CXX_TARGETS := $(CXX_TARGETS:%.cc=%.x)

CUDA_TARGETS = $(shell find . -maxdepth 1 -name '*cuda.cu')
CUDA_TARGETS := $(CUDA_TARGETS:%cuda.cu=%cuda.x)

CUSTATEVEC_TARGETS = $(shell find . -maxdepth 1 -name "*custatevec.cu")
CUSTATEVEC_TARGETS := $(CUSTATEVEC_TARGETS:%custatevec.cu=%custatevec.x)

.PHONY: qsim
qsim: $(CXX_TARGETS)

.PHONY: qsim-cuda
qsim-cuda: $(CUDA_TARGETS)

.PHONY: qsim-custatevec
qsim-custatevec: $(CUSTATEVEC_TARGETS)

%.x: %.cc
	$(CXX) -o ./$@ $< $(CXXFLAGS) $(ARCHFLAGS)

%cuda.x: %cuda.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS)

%custatevec.x: %custatevec.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS) $(CUSTATEVECFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
