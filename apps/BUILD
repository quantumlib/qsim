cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    deps = [
        ":getopt",
        "//lib:run_qsim_lib",
    ],
    data = ["//circuits:circuit_q24"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    deps = [
        ":getopt",
        "//lib:run_qsim_lib",
    ],
)

cc_library(
    name = "getopt",
    srcs = [
        "getopt.cc",
        "getopt.h",
    ],
    hdrs = ["getopt.h"],
    deps = [],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    deps = [
        ":getopt",
        "//lib:bitstring",
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    deps = [
        ":getopt",
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    deps = [
        "getopt",
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)