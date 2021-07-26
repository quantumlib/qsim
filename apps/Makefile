CXX_TARGETS = $(shell find . -maxdepth 1 -name '*.cc')
CXX_TARGETS := $(CXX_TARGETS:%.cc=%.x)

CUDA_TARGETS = $(shell find . -maxdepth 1 -name '*.cu')
CUDA_TARGETS := $(CUDA_TARGETS:%.cu=%.x)

.PHONY: qsim
qsim: $(CXX_TARGETS)

.PHONY: qsim-cuda
qsim-cuda: $(CUDA_TARGETS)

%.x: %.cc
	$(CXX) -o ./$@ $< $(CXXFLAGS) $(ARCHFLAGS)

%.x: %.cu
	$(NVCC) -o ./$@ $< $(NVCCFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
