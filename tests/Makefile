TARGETS = $(shell find . -maxdepth 1 -name '*_test.cc')
TARGETS := $(TARGETS:%.cc=%.x)

GTEST_DIR = $(CURDIR)/googletest/googletest
GMOCK_DIR = $(CURDIR)/googletest/googlemock

CMAKE=cmake

TESTFLAGS = -I$(GTEST_DIR)/include -L$(GTEST_DIR)/make/lib -lpthread -lgtest

.PHONY: all
all: $(TARGETS)

.PHONE: run-all
run-all: all
	for exe in $(TARGETS); do if ! ./$$exe; then exit 1; fi; done

$(GTEST_DIR)/make:
	mkdir -p $(GTEST_DIR)/make
	cd $(GTEST_DIR)/make &&	$(CMAKE) .. && $(MAKE)

%.x: %.cc $(GTEST_DIR)/make
	$(CXX) -o ./$@ $< $(TESTFLAGS) $(CXXFLAGS)

.PHONY: clean
clean:
	-rm -f ./*.x ./*.a ./*.so ./*.mod
	$(MAKE) -C $(GTEST_DIR) clean
	rm -rf $(GTEST_DIR)/make