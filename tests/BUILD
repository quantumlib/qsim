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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

# Options for testing different simulator types.
avx_copts = [
    "-mavx2",
    "-mfma",
]

avx512_copts = ["-march=native"]

sse_copts = ["-msse4"]

windows_copts = [
    "/arch:AVX",
    "/std:c++17",
]

windows_avx512_copts = [
    "/arch:AVX512",
    "/std:c++17",
]

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

cc_test(
    name = "bitstring_test",
    size = "small",
    srcs = ["bitstring_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:bitstring",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "channel_test",
    size = "small",
    srcs = ["channel_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:channel",
        "//lib:formux",
        "//lib:gates_cirq",
        "//lib:matrix",
        "//lib:simulator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "channels_cirq_test",
    size = "small",
    srcs = ["channels_cirq_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:channels_cirq",
        "//lib:circuit",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gates_cirq",
        "//lib:io",
        "//lib:qtrajectory",
        "//lib:run_qsim",
        "//lib:simulator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "circuit_qsim_parser_test",
    size = "small",
    srcs = ["circuit_qsim_parser_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:gates_qsim",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "expect_test",
    size = "small",
    srcs = ["expect_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:expect",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_qsim",
        "//lib:io",
        "//lib:simulator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fuser_basic_test",
    size = "small",
    srcs = ["fuser_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fuser_mqubit_test",
    size = "small",
    srcs = ["fuser_mqubit_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gate",
        "//lib:gate_appl",
        "//lib:matrix",
        "//lib:simulator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gates_cirq_testfixture",
    testonly = 1,
    hdrs = ["gates_cirq_testfixture.h"],
    deps = [
        "//lib:circuit",
        "//lib:gates_cirq",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gates_qsim_test",
    size = "small",
    srcs = ["gates_qsim_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:gates_qsim",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hybrid_testfixture",
    testonly = 1,
    hdrs = ["hybrid_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:formux",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:hybrid",
        "//lib:io",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hybrid_avx_test",
    size = "small",
    srcs = ["hybrid_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":hybrid_testfixture",
        "//lib:seqfor",
        "//lib:simulator_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "matrix_test",
    size = "small",
    srcs = ["matrix_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:matrix",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "qtrajectory_testfixture",
    testonly = 1,
    hdrs = ["qtrajectory_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:channel",
        "//lib:channels_cirq",
        "//lib:circuit_noisy",
        "//lib:expect",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_cirq",
        "//lib:io",
        "//lib:qtrajectory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "qtrajectory_avx_test",
    size = "small",
    srcs = ["qtrajectory_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":qtrajectory_testfixture",
        "//lib:fuser_mqubit",
        "//lib:gates_cirq",
        "//lib:io",
        "//lib:run_qsim",
        "//lib:seqfor",
        "//lib:simulator_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "run_qsim_test",
    size = "small",
    srcs = ["run_qsim_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":gates_cirq_testfixture",
        "//lib:run_qsim_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "run_qsimh_test",
    size = "small",
    srcs = ["run_qsimh_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":gates_cirq_testfixture",
        "//lib:run_qsimh_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simulator_testfixture",
    testonly = 1,
    hdrs = ["simulator_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:expect",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_qsim",
        "//lib:io",
        "//lib:util_cpu",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simulator_avx_test",
    size = "small",
    srcs = ["simulator_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":simulator_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simulator_avx512_test",
    size = "small",
    srcs = ["simulator_avx512_test.cc"],
    copts = select({
        ":windows": windows_avx512_copts,
        "//conditions:default": avx512_copts,
    }),
    tags = ["avx"],
    deps = [
        ":simulator_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx512",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simulator_basic_test",
    size = "small",
    srcs = ["simulator_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":simulator_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_basic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "simulator_sse_test",
    size = "small",
    srcs = ["simulator_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    tags = ["sse"],
    deps = [
        ":simulator_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_sse",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "statespace_testfixture",
    testonly = 1,
    hdrs = ["statespace_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io",
        "//lib:run_qsim",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "statespace_avx_test",
    size = "small",
    srcs = ["statespace_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":statespace_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx",
        "//lib:statespace_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "statespace_avx512_test",
    size = "small",
    srcs = ["statespace_avx512_test.cc"],
    copts = select({
        ":windows": windows_avx512_copts,
        "//conditions:default": avx512_copts,
    }),
    tags = ["avx"],
    deps = [
        ":statespace_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx512",
        "//lib:statespace_avx512",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "statespace_basic_test",
    size = "small",
    srcs = ["statespace_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":statespace_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_basic",
        "//lib:statespace_basic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "statespace_sse_test",
    size = "small",
    srcs = ["statespace_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    tags = ["sse"],
    deps = [
        ":statespace_testfixture",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_sse",
        "//lib:statespace_sse",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "unitaryspace_testfixture",
    testonly = 1,
    hdrs = ["unitaryspace_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitaryspace_avx_test",
    size = "small",
    srcs = ["unitaryspace_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":unitaryspace_testfixture",
        "//lib:formux",
        "//lib:unitaryspace_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitaryspace_avx512_test",
    size = "small",
    srcs = ["unitaryspace_avx512_test.cc"],
    copts = select({
        ":windows": windows_avx512_copts,
        "//conditions:default": avx512_copts,
    }),
    tags = ["avx"],
    deps = [
        ":unitaryspace_testfixture",
        "//lib:formux",
        "//lib:unitaryspace_avx512",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitaryspace_basic_test",
    size = "small",
    srcs = ["unitaryspace_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":unitaryspace_testfixture",
        "//lib:formux",
        "//lib:unitaryspace_basic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitaryspace_sse_test",
    size = "small",
    srcs = ["unitaryspace_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    tags = ["sse"],
    deps = [
        ":unitaryspace_testfixture",
        "//lib:formux",
        "//lib:unitaryspace_sse",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "unitary_calculator_testfixture",
    testonly = 1,
    hdrs = ["unitary_calculator_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:fuser",
        "//lib:gate_appl",
        "//lib:gates_cirq",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitary_calculator_avx_test",
    size = "small",
    srcs = ["unitary_calculator_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    tags = ["avx"],
    deps = [
        ":unitary_calculator_testfixture",
        "//lib:formux",
        "//lib:unitary_calculator_avx",
        "//lib:unitaryspace_avx",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitary_calculator_avx512_test",
    size = "small",
    srcs = ["unitary_calculator_avx512_test.cc"],
    copts = select({
        ":windows": windows_avx512_copts,
        "//conditions:default": avx512_copts,
    }),
    tags = ["avx"],
    deps = [
        ":unitary_calculator_testfixture",
        "//lib:formux",
        "//lib:unitary_calculator_avx512",
        "//lib:unitaryspace_avx512",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitary_calculator_basic_test",
    size = "small",
    srcs = ["unitary_calculator_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":unitary_calculator_testfixture",
        "//lib:formux",
        "//lib:unitary_calculator_basic",
        "//lib:unitaryspace_basic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "unitary_calculator_sse_test",
    size = "small",
    srcs = ["unitary_calculator_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    tags = ["sse"],
    deps = [
        ":unitary_calculator_testfixture",
        "//lib:formux",
        "//lib:unitary_calculator_sse",
        "//lib:unitaryspace_sse",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "vectorspace_test",
    size = "small",
    srcs = ["vectorspace_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:formux",
        "//lib:vectorspace",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mps_statespace_test",
    size = "small",
    srcs = ["mps_statespace_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:formux",
        "//lib:mps_statespace",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mps_simulator_test",
    size = "small",
    srcs = ["mps_simulator_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:formux",
        "//lib:gate_appl",
        "//lib:gates_cirq",
        "//lib:gates_qsim",
        "//lib:mps_simulator",
        "@com_google_googletest//:gtest_main",
    ],
)
