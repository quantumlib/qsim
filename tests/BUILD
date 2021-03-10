# Options for testing different simulator types.
avx_copts = ['-mavx2', '-mfma']
sse_copts = ['-msse4']

windows_copts = [
    "/arch:AVX",
    "/std:c++14",
]

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_test(
    name = "bitstring_test",
    srcs = ["bitstring_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:bitstring",
    ],
)

cc_test(
    name = "channels_cirq_test",
    srcs = ["channels_cirq_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:channels_cirq",
        "//lib:circuit",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gates_cirq",
        "//lib:io",
        "//lib:qtrajectory",
        "//lib:simulator",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "circuit_qsim_parser_test",
    srcs = ["circuit_qsim_parser_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:gates_qsim",
    ],
)

cc_test(
    name = "expect_test",
    srcs = ["expect_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:expect",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_qsim",
        "//lib:io",
        "//lib:simulator",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "fuser_basic_test",
    srcs = ["fuser_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
    ],
)

cc_test(
    name = "fuser_mqubit_test",
    srcs = ["fuser_mqubit_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gate",
        "//lib:gate_appl",
        "//lib:matrix",
        "//lib:simulator",
    ],
)

cc_library(
    name = "gates_cirq_testfixture",
    hdrs = ["gates_cirq_testfixture.h"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit",
        "//lib:gates_cirq",
    ],
    testonly = 1,
)

cc_test(
    name = "gates_qsim_test",
    srcs = ["gates_qsim_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:gates_qsim",
    ],
)

cc_test(
    name = "hybrid_test",
    srcs = ["hybrid_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:formux",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:hybrid",
        "//lib:io",
        "//lib:simulator",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:matrix",
    ],
)

cc_test(
    name = "qtrajectory_test",
    srcs = ["qtrajectory_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:channels_cirq",
        "//lib:circuit_noisy",
        "//lib:formux",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_cirq",
        "//lib:io",
        "//lib:qtrajectory",
        "//lib:simulator",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "run_qsim_test",
    srcs = ["run_qsim_test.cc"],
    deps = [
        ":gates_cirq_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:run_qsim_lib",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "run_qsimh_test",
    srcs = ["run_qsimh_test.cc"],
    deps = [
        ":gates_cirq_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:run_qsimh_lib",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "simulator_testfixture",
    hdrs = ["simulator_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:expect",
        "//lib:fuser_mqubit",
        "//lib:gate_appl",
        "//lib:gates_qsim",
        "//lib:io",
    ],
    testonly = 1,
)

cc_test(
    name = "simulator_avx_test",
    srcs = ["simulator_avx_test.cc"],
    deps = [
        ":simulator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
)

cc_test(
    name = "simulator_basic_test",
    srcs = ["simulator_basic_test.cc"],
    deps = [
        ":simulator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_basic",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "simulator_sse_test",
    srcs = ["simulator_sse_test.cc"],
    deps = [
        ":simulator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_sse",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
)

cc_library(
    name = "statespace_testfixture",
    hdrs = ["statespace_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io",
        "//lib:run_qsim",
    ],
    testonly = 1,
)

cc_test(
    name = "statespace_avx_test",
    srcs = ["statespace_avx_test.cc"],
    deps = [
        ":statespace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_avx",
        "//lib:statespace_avx",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
)

cc_test(
    name = "statespace_basic_test",
    srcs = ["statespace_basic_test.cc"],
    deps = [
        ":statespace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_basic",
        "//lib:statespace_basic",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
)

cc_test(
    name = "statespace_sse_test",
    srcs = ["statespace_sse_test.cc"],
    deps = [
        ":statespace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:seqfor",
        "//lib:simulator_sse",
        "//lib:statespace_sse",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
)

cc_library(
    name = "unitaryspace_testfixture",
    hdrs = ["unitaryspace_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
    testonly = 1,
)

cc_test(
    name = "unitaryspace_avx_test",
    srcs = ["unitaryspace_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    deps = [
        ":unitaryspace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_avx"
    ],
)

cc_test(
    name = "unitaryspace_basic_test",
    srcs = ["unitaryspace_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":unitaryspace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_basic"
    ],
)

cc_test(
    name = "unitaryspace_sse_test",
    srcs = ["unitaryspace_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    deps = [
        ":unitaryspace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_sse"
    ],
)

cc_library(
    name = "unitary_calculator_testfixture",
    hdrs = ["unitary_calculator_testfixture.h"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:bits",
        "//lib:fuser",
        "//lib:gate_appl",
        "//lib:gates_cirq",
    ],
    testonly = 1,
)

cc_test(
    name = "unitary_calculator_avx_test",
    srcs = ["unitary_calculator_avx_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts,
    }),
    deps = [
        ":unitary_calculator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_avx",
        "//lib:unitary_calculator_avx",
    ],
)

cc_test(
    name = "unitary_calculator_basic_test",
    srcs = ["unitary_calculator_basic_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        ":unitary_calculator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_basic",
        "//lib:unitary_calculator_basic",
    ],
)

cc_test(
    name = "unitary_calculator_sse_test",
    srcs = ["unitary_calculator_sse_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts,
    }),
    deps = [
        ":unitary_calculator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:unitaryspace_sse",
        "//lib:unitary_calculator_sse",
    ],
)

cc_test(
    name = "vectorspace_test",
    srcs = ["vectorspace_test.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:formux",
        "//lib:vectorspace",
    ],
)
