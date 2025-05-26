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
shopt -s inherit_errexit

declare -r usage="Usage: ${0##*/} [-h | --help | help] [bazel options ...]
Run the programs in tests/, and on Linux, also build the programs in apps/.

If the first option on the command line is -h, --help, or help, this help text
will be printed and the program will exit. Any other options on the command
line are passed directly to Bazel.

Note: the MacOS VMs in GitHub runners may run on different-capability CPUS, so
all AVX versions of programs in tests/ are automatically excluded."

# Exit early if the user requested help.
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    echo "$usage"
    exit 0
fi

declare filter_avx=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    filter_avx="--test_tag_filters=-avx"
fi

# Apps are sample programs and are only meant to run on Linux.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    bazel build --config=sse $filter_avx "$@" apps:all
    bazel build $filter_avx "$@" apps:all
fi

# Run all basic tests. This should work on all platforms.
bazel test $filter_avx "$@" tests:all
