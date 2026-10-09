CXX_TARGETS = $(shell find . -maxdepth 1 -name '*.cc')
CXX_TARGETS := $(CXX_TARGETS:%.cc=%.x)

OUTPUT_DIR ?= .

CUDA_TARGETS = $(shell find . -maxdepth 1 -name '*cuda.cu')
CUDA_TARGETS := $(CUDA_TARGETS:%cuda.cu=%cuda.x)

CUSTATEVEC_TARGETS = $(shell find . -maxdepth 1 -name "*custatevec.cu")
CUSTATEVEC_TARGETS := $(CUSTATEVEC_TARGETS:%custatevec.cu=%custatevec.x)

CUSTATEVECEX_TARGETS = $(shell find . -maxdepth 1 -name "*custatevecex.cu")
CUSTATEVECEX_TARGETS := $(CUSTATEVECEX_TARGETS:%custatevecex.cu=%custatevecex.x)

HIP_TARGETS = $(shell find . -maxdepth 1 -name '*cuda.cu')
HIP_TARGETS := $(HIP_TARGETS:%cuda.cu=%hip.x)

.PHONY: qsim
qsim: $(CXX_TARGETS)

.PHONY: gate-batch
gate-batch:
	mkdir -p "$(OUTPUT_DIR)"
	$(CXX) -o "$(OUTPUT_DIR)/qsim_gate_batch.x" qsim_gate_batch.cc $(CXXFLAGS) $(ARCHFLAGS)
	$(CXX) -o "$(OUTPUT_DIR)/qsim_base.x" qsim_base.cc $(CXXFLAGS) $(ARCHFLAGS)

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
