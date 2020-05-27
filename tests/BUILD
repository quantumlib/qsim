# Options for testing different simulator types.
avx_copts = ['-mavx2', '-mfma']
sse_copts = ['-msse4']
omp_copts = ['-fopenmp']
omp_linkopts = ['-lgomp']

windows_copts = [
    "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
    "/D_USE_MATH_DEFINES",
    "/DEIGEN_MAX_ALIGN_BYTES=64",
    "/DEIGEN_HAS_TYPE_TRAITS=0",
    "/DTF_USE_SNAPPY",
    "/showIncludes",
    "/MD",
    "/O2",
    "/DNDEBUG",
    "/w",
    "-DWIN32_LEAN_AND_MEAN",
    "-DNOGDI",
    "/d2ReducedOptimizeHugeFunctions",
    "/arch:AVX",
    "/std:c++14",
    "/DPLATFORM_WINDOWS",
    "/DEIGEN_HAS_C99_MATH",
    "/DEIGEN_AVOID_STL_ARRAY",
    "/Iexternal/gemmlowp",
    "/wd4018",
    "/wd4577",
    "/DNOGDI",
    "/UTF_COMPILE_LIBRARY",
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
        "//lib:io",
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
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:hybrid",
        "//lib:io",
        "//lib:parfor",
        "//lib:simulator_avx",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
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
    name = "run_qsim_test",
    srcs = ["run_qsim_test.cc"],
    deps = [
        ":gates_cirq_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:run_qsim_lib",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
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
        "//conditions:default": avx_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
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
        "//lib:fuser",
        "//lib:gate_appl",
        "//lib:gates_qsim",
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
        "//lib:simulator_avx",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
)

cc_test(
    name = "simulator_basic_test",
    srcs = ["simulator_basic_test.cc"],
    deps = [
        ":simulator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:simulator_basic",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": omp_copts,
    }),
    linkopts = omp_linkopts,
)

cc_test(
    name = "simulator_sse_test",
    srcs = ["simulator_sse_test.cc"],
    deps = [
        ":simulator_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:simulator_sse",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
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
        "//lib:simulator_avx",
        "//lib:statespace_avx",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": avx_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
)

cc_test(
    name = "statespace_basic_test",
    srcs = ["statespace_basic_test.cc"],
    deps = [
        ":statespace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:simulator_basic",
        "//lib:statespace_basic",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": omp_copts,
    }),
    linkopts = omp_linkopts,
)

cc_test(
    name = "statespace_sse_test",
    srcs = ["statespace_sse_test.cc"],
    deps = [
        ":statespace_testfixture",
        "@com_google_googletest//:gtest_main",
        "//lib:parfor",
        "//lib:simulator_sse",
        "//lib:statespace_sse",
    ],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": sse_copts + omp_copts,
    }),
    linkopts = omp_linkopts,
)
