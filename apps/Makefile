TARGETS = $(shell find . -maxdepth 1 -name '*.cc')
TARGETS := $(TARGETS:%.cc=%.x)

.PHONY: all
all: $(TARGETS)

%.x: %.cc
	$(CXX) -o ./$@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod