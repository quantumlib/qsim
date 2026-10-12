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

# Requires google-benchmark (https://github.com/google/benchmark) to be
# installed, e.g. `apt install libbenchmark-dev`.

CXXFLAGS ?= -O3 -std=c++17 -DNDEBUG
BENCHFLAGS = -lbenchmark -lpthread

BENCH_FILES := $(wildcard *_benchmark.cc)
BENCH_TARGETS := $(BENCH_FILES:%.cc=%.x)

.PHONY: benchmarks
benchmarks: $(BENCH_TARGETS)

.PHONY: run-benchmarks
run-benchmarks: benchmarks
	for exe in $(BENCH_TARGETS); do ./$$exe; done

%.x: %.cc
	$(CXX) -o ./$@ $< $(CXXFLAGS) $(BENCHFLAGS)

.PHONY: clean
clean:
	-rm -f *.x
