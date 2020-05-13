package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bitstring",
    hdrs = ["bitstring.h"],
)

cc_library(
    name = "circuit",
    hdrs = ["circuit.h"],
)

cc_library(
    name = "fuser",
    hdrs = ["fuser.h"],
)

cc_library(
    name = "io",
    hdrs = ["io.h"],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
)

cc_library(
    name = "parfor",
    hdrs = ["parfor.h"],
)

cc_library(
    name = "seqfor",
    hdrs = ["seqfor.h"],
)

cc_library(
    name = "statespace",
    hdrs = ["statespace.h"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
)

cc_library(
    name = "fuser_basic",
    hdrs = ["fuser_basic.h"],
    deps = [":fuser"],
)

cc_library(
    name = "gate",
    hdrs = ["gate.h"],
    deps = [":matrix"],
)

cc_library(
    name = "gates_appl",
    hdrs = ["gates_appl.h"],
    deps = [":matrix"],
)

cc_library(
    name = "gates_qsim",
    hdrs = ["gates_qsim.h"],
    deps = [":gate"],
)

cc_library(
    name = "circuit_qsim_parser",
    hdrs = ["circuit_qsim_parser.h"],
    deps = [
        ":circuit",
        ":gates_qsim",
    ],
)

cc_library(
    name = "hybrid",
    hdrs = ["hybrid.h"],
    deps = [
        ":gate",
        ":gates_appl",
    ],
)

cc_library(
    name = "io_file",
    hdrs = ["io_file.h"],
    deps = [":io"],
)

cc_library(
    name = "run_qsim",
    hdrs = ["run_qsim.h"],
    deps = [
        ":gates_appl",
        ":util",
    ],
)

cc_library(
    name = "run_qsimh",
    hdrs = ["run_qsimh.h"],
    deps = [
        ":hybrid",
        ":util",
    ],
)

cc_library(
    name = "statespace_avx",
    hdrs = ["statespace_avx.h"],
    deps = [
        ":statespace",
        ":util",
    ],
)

cc_library(
    name = "statespace_basic",
    hdrs = ["statespace_basic.h"],
    deps = [
        ":statespace",
        ":util",
    ],
)

cc_library(
    name = "statespace_sse",
    hdrs = ["statespace_sse.h"],
    deps = [
        ":statespace",
        ":util",
    ],
)

cc_library(
    name = "simulator_avx",
    hdrs = ["simulator_avx.h"],
    deps = [":statespace_avx"],
)

cc_library(
    name = "simulator_basic",
    hdrs = ["simulator_basic.h"],
    deps = [":statespace_basic"],
)

cc_library(
    name = "simulator_sse",
    hdrs = ["simulator_sse.h"],
    deps = [":statespace_sse"],
)

cc_library(
    name = "simulator",
    hdrs = ["simmux.h"],
    deps = [
        ":simulator_avx",
        ":simulator_basic",
        ":simulator_sse",
    ],
)
