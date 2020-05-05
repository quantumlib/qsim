package(default_visibility = ["//visibility:public"])

##### Aggregate libraries #####

# Full qsim library
cc_library(
    name = "qsim_lib",
    hdrs = [
        "bitstring.h",
        "circuit_qsim_parser.h",
        "circuit.h",
        "fuser_basic.h",
        "fuser.h",
        "gate.h",
        "gates_appl.h",
        "gates_qsim.h",
        "hybrid.h",
        "io_file.h",
        "io.h",
        "matrix.h",
        "parfor.h",
        "run_qsim.h",
        "run_qsimh.h",
        "seqfor.h",
        "simmux.h",
        "simulator_avx.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace_avx.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "statespace.h",
        "util.h",
    ],
)

# Library to run qsim with qsim circuit parser and parallel `for`
cc_library(
    name = "run_qsim_lib",
    deps = [
        ":circuit_qsim_parser",
        ":fuser_basic",
        ":gates_qsim",
        ":io_file",
        ":parfor",
        ":run_qsim",
        ":simulator",
    ],
)

# Library to run qsimh with qsim circuit parser and parallel `for`
cc_library(
    name = "run_qsimh_lib",
    deps = [
        ":circuit_qsim_parser",
        ":fuser_basic",
        ":gates_qsim",
        ":io_file",
        ":parfor",
        ":run_qsimh",
        ":simulator",
    ],
)

##### Basic libraries #####

### Utility libraries ###

# Bitstring parser
cc_library(
    name = "bitstring",
    hdrs = ["bitstring.h"],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
)

### Input/output libraries ###

cc_library(
    name = "io",
    hdrs = ["io.h"],
)

cc_library(
    name = "io_file",
    hdrs = ["io_file.h"],
    deps = [":io"],
)

### Parallel and sequential `for` libraries ###

# OpenMP-based parallelization
cc_library(
    name = "parfor",
    hdrs = ["parfor.h"],
    copts = ["-fopenmp"],
)

cc_library(
    name = "seqfor",
    hdrs = ["seqfor.h"],
)

### Gate libraries ###

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

### Circuit libraries ###

cc_library(
    name = "circuit",
    hdrs = ["circuit.h"],
)

# qsim circuit parser
cc_library(
    name = "circuit_qsim_parser",
    hdrs = ["circuit_qsim_parser.h"],
    deps = [
        ":circuit",
        ":gates_qsim",
    ],
)

### Fuser libraries ###

cc_library(
    name = "fuser",
    hdrs = ["fuser.h"],
)

cc_library(
    name = "fuser_basic",
    hdrs = ["fuser_basic.h"],
    deps = [":fuser"],
)

### Helper libraries to run qsim and qsimh ###

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

### Statespace libraries ###

cc_library(
    name = "statespace",
    hdrs = ["statespace.h"],
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

### Simulator libraries ###

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

# All three state-vector simulators with multiplexer
cc_library(
    name = "simulator",
    hdrs = ["simmux.h"],
    deps = [
        ":simulator_avx",
        ":simulator_basic",
        ":simulator_sse",
    ],
)

# Hybrid simulator
cc_library(
    name = "hybrid",
    hdrs = ["hybrid.h"],
    deps = [
        ":gate",
        ":gates_appl",
    ],
)
