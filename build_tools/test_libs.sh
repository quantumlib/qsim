#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

which bazel
bazel version

# Attempt to build all components in SSE and basic mode.
# The Github Action MacOS VMs may run on different-capability CPUs, so all AVX tests
# are excluded from the build and test process.
# apps are sample applications for experts and are meant to be run on only on Linux.

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  bazel build --config=sse apps:all
  bazel build apps:all
fi

# Run all basic tests.
set e  # Ignore errors until artifacts are collected.
EXIT_CODE=0
for TARGET in bitstring_test channels_cirq_test circuit_qsim_parser_test expect_test \
              fuser_basic_test gates_qsim_test hybrid_test matrix_test qtrajectory_test \
              run_qsim_test run_qsimh_test simulator_basic_test simulator_sse_test statespace_basic_test \
              statespace_sse_test unitary_calculator_basic_test unitary_calculator_sse_test \
              unitaryspace_basic_test unitaryspace_sse_test vectorspace_test; do \
  if ! bazel test --test_output=errors tests:${TARGET}; then
    EXIT_CODE=1
  fi
done
