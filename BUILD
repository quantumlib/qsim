# Copyright 2026 Google LLC
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

load("@local_compiler_config//:compiler_config.bzl", "AVX_COPTS", "SSE_COPTS")
load("//dev_tools:bazel_utils.bzl", "qsim_print_flags", "qsim_select_copts")

qsim_print_flags(
    name = "show_flags",
    flags = qsim_select_copts(
        avx_copts = AVX_COPTS,
        default = [],
        native_copts = ["-march=native"],
        sse_copts = SSE_COPTS,
        windows_copts = AVX_COPTS,
    ),
    visibility = ["//visibility:public"],
)

# Define configurations for build with AVX and/or SSE support.

config_setting(
    name = "avx_requested",
    values = {"define": "qsim_avx=true"},
)

config_setting(
    name = "sse_requested",
    values = {"define": "qsim_sse=true"},
)

config_setting(
    name = "avx_and_sse_requested",
    define_values = {
        "qsim_avx": "true",
        "qsim_sse": "true",
    },
)

config_setting(
    name = "native_requested",
    values = {"define": "qsim_native=true"},
)
