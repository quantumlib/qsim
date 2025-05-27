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

# Unless we can tell this system supports AVX, we skip those tests.
declare filter_avx="--build_tag_filters=-avx --test_tag_filters=-avx"
declare filter_sse="--build_tag_filters=-sse --test_tag_filters=-sse"
declare config_sse=""
shopt -s nocasematch
# Note: can't use Bash's $OSTYPE here b/c the value is "linux-gnu" on Win 10.
case "$(uname -s)" in
    darwin*)
        features=$(sysctl machdep.cpu.features)
        [[ "$features" == *"AVX2"* ]] && filter_avx=""
        [[ "$features" == *"sse"* ]] && filter_sse=""
        ;;
    linux*)
        if grep -qi flags /proc/cpuinfo | grep -qi "avx2"; then
            filter_avx=""
            config_sse="--config=sse"
        fi
        if grep -qi flags /proc/cpuinfo | grep -qi "sse"; then
            filter_sse=""
            config_sse="--config=sse"
        fi
        ;;
    windows*|cygwin*|mingw32*|msys*|mingw*)
        if wmic cpu get Caption /value | grep -qi "avx2"; then
            filter_avx=""
        elif wmic cpu get InstructionSet /value | grep -qi "avx2"; then
            filter_avx=""
        fi
        if wmic cpu get Caption /value | grep -qi "sse"; then
            filter_sse=""
        elif wmic cpu get InstructionSet /value | grep -qi "sse"; then
            filter_sse=""
        fi
        ;;
esac

# Apps are sample programs and are only meant to run on Linux.
# shellcheck disable=SC2086
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    bazel build $filter_avx $filter_sse $config_sse "$@" apps:all
    bazel build $filter_avx $filter_sse "$@" apps:all
fi

# Run all basic tests. This should work on all platforms.
# shellcheck disable=SC2086
bazel test $filter_avx $filter_sse "$@" tests:all
