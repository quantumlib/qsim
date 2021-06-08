#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

which bazel
bazel version

# Attempt to build all components in SSE and basic mode.
# The Kokoro MacOS VMs are not configured for AVX2 or OpenMP, so these modes
# are excluded from the build and test process.
#bazel build --config=sse apps:all
bazel build --config=avx apps:all
#bazel build apps:all

# Run all basic tests.
bazel test --test_output=errors tests:all