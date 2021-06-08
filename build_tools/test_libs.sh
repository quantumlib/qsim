#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

which bazel
bazel version

# Attempt to build all components in SSE and basic mode.
# The Github Actions MacOS VMs are not configured for AVX2 or OpenMP, so these modes
# are excluded from the build and test process.
bazel build --config=sse apps:all
bazel build apps:all

# Run all basic tests.
set +e  # Ignore errors until artifacts are collected.
EXIT_CODE=0
for TARGET in bitstring_test circuit_qsim_parser_test fuser_basic_test \
              gates_qsim_test hybrid_test matrix_test run_qsim_test \
              run_qsimh_test simulator_basic_test statespace_basic_test; do \
  if ! bazel test --test_output=errors tests:${TARGET}; then
    EXIT_CODE=1
  fi
done