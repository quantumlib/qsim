# Options for testing different simulator types.
avx_copts = ['-mavx2', '-mfma']
sse_copts = ['-msse4']
omp_copts = ['-fopenmp']
omp_linkopts = ['-lgomp']

cc_test(
    name = "bitstring_test",
    srcs = ["bitstring_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:bitstring",
        "//lib:io",
    ],
)

cc_test(
    name = "circuit_qsim_parser_test",
    srcs = ["circuit_qsim_parser_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:gates_qsim",
        "//lib:io",
    ],
)

cc_test(
    name = "fuser_basic_test",
    srcs = ["fuser_basic_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io",
    ],
)

cc_test(
    name = "gates_qsim_test",
    srcs = ["gates_qsim_test.cc"],
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
    copts = avx_copts + omp_copts,
    linkopts = omp_linkopts,
)

cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:matrix",
    ],
)

cc_test(
    name = "run_qsim_test",
    srcs = ["run_qsim_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:run_qsim_lib",
    ],
    copts = avx_copts + omp_copts,
    linkopts = omp_linkopts,
)

cc_test(
    name = "run_qsimh_test",
    srcs = ["run_qsimh_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:run_qsimh_lib",
    ],
    copts = avx_copts + omp_copts,
    linkopts = omp_linkopts,
)

cc_library(
    name = "simulator_testfixture",
    hdrs = ["simulator_testfixture.h"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//lib:fuser",
        "//lib:gates_appl",
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
    copts = avx_copts + omp_copts,
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
    copts = omp_copts,
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
    copts = sse_copts + omp_copts,
    linkopts = omp_linkopts,
)

cc_library(
    name = "statespace_testfixture",
    hdrs = ["statespace_testfixture.h"],
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
    copts = avx_copts + omp_copts,
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
    copts = omp_copts,
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
    copts = sse_copts + omp_copts,
    linkopts = omp_linkopts,
)
