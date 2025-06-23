#!/usr/bin/env bash
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

set -eo pipefail -o errtrace

declare -r usage="Usage: ${0##*/} [-h | --help | help] [bazel options ...]
Run the programs in tests/, and on Linux, also build the programs in apps/.

If the first option on the command line is -h, --help, or help, this help text
will be printed and the program will exit. Any other options on the command
line are passed directly to Bazel."

# Exit early if the user requested help.
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    echo "$usage"
    exit 0
fi

# Look for AVX and SSE in the processor's feature flags.
declare features=""
declare filters=""
declare -a configs=()
features="$(python -c 'import cpuinfo; print(" ".join(cpuinfo.get_cpu_info().get("flags", [])))')"
if [[ "$features" == *avx2* ]]; then
     filters+=",avx"
     configs+=( "--config=avx" )
fi
if [[ "$features" == *sse* ]]; then
     filters+=",sse"
     configs+=( "--config=sse" )
fi
filters="${filters#,}"

# If none of the optimization configs were added, use the basic config.
if [[ ${#configs[@]} -eq 0 ]]; then
    configs=( "--config=basic" )
fi

declare -a build_filters=()
declare -a test_filters=()
if [[ -n "$filters" ]]; then
    build_filters=( "--build_tag_filters=$filters" )
    test_filters=( "--test_tag_filters=$filters" )
fi

# The apps are sample programs and are only meant to run on Linux.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    bazel build "${configs[@]}" "${build_filters[@]}" "$@" apps:all
fi

# Run all basic tests. This should work on all platforms.
bazel test "${configs[@]}" "${test_filters[@]}" "$@" tests:all
