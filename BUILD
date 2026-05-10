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

load("//dev_tools:bazel_utils.bzl", "qsim_print_flags", "qsim_select_copts")

qsim_print_flags(
    name = "show_flags",
    flags = qsim_select_copts(target_level = "avx"),
    visibility = ["//visibility:public"],
)

# Define configurations for build with AVX and/or SSE support.

config_setting(
    name = "avx_requested",
    values = {"define": "qsim_avx=true"},
)

config_setting(
    name = "avx2_requested",
    values = {"define": "qsim_avx2=true"},
)

config_setting(
    name = "avx512_requested",
    values = {"define": "qsim_avx512=true"},
)

config_setting(
    name = "avx_and_avx2_requested",
    define_values = {
        "qsim_avx": "true",
        "qsim_avx2": "true",
    },
)

config_setting(
    name = "avx_and_avx512_requested",
    define_values = {
        "qsim_avx": "true",
        "qsim_avx512": "true",
    },
)

config_setting(
    name = "avx512_and_avx2_requested",
    define_values = {
        "qsim_avx512": "true",
        "qsim_avx2": "true",
    },
)

config_setting(
    name = "avx_all_requested",
    define_values = {
        "qsim_avx": "true",
        "qsim_avx2": "true",
        "qsim_avx512": "true",
    },
)

config_setting(
    name = "avx_and_sse_requested",
    define_values = {
        "qsim_avx": "true",
        "qsim_sse": "true",
    },
)

config_setting(
    name = "sse_requested",
    values = {"define": "qsim_sse=true"},
)

config_setting(
    name = "native_requested",
    values = {"define": "qsim_native=true"},
)
