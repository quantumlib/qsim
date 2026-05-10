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

Invokes Bazel to run the programs in tests/, and on Linux, also build the
sample programs in apps/.

If the first option on the command line is -h, --help, or help, this help text
will be printed and the program will exit. Any other options on the command
line are passed directly to Bazel."

# Exit early if the user requested help.
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    echo "$usage"
    exit 0
fi

# We use the 'native' config to automatically detect and use the best 
# instruction set available on the current host.
declare -a configs=( "--config=native" )

# The apps are sample programs and are only meant to run on Linux.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    bazel build "${configs[@]}" "$@" apps:all
fi

# Run all tests. Incompatible SIMD tests will be skipped automatically.
bazel test "${configs[@]}" "$@" tests:all
