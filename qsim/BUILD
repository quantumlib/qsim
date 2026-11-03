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

load("@local_compiler_config//:compiler_config.bzl", "SUPPORTS_GSFRAME")
load("@rules_cc//cc:defs.bzl", "cc_binary")

gsframe_copts = select({
    "@platforms//os:linux": ["-Wa,--gsframe=no"] if SUPPORTS_GSFRAME else [],
    "//conditions:default": [],
})

copts = [
    "-flto=auto",
    "-fopenmp",
    "-march=native",
    "-O3",
    "-std=c++20",
] + gsframe_copts

cc_binary(
    name = "qsim",
    srcs = ["qsim.cc"],
    copts = copts,
    linkopts = [
        "-lgomp",
        "-flto=auto",
    ],
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:classical_control",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:io",
        "//lib:run_qsim",
        "//lib:simulator",
        "//lib:util",
        "//lib:util_cpu",
    ],
)
