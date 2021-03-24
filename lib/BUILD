package(default_visibility = ["//visibility:public"])

##### Aggregate libraries #####

# Full qsim library
cc_library(
    name = "qsim_lib",
    hdrs = [
        "bits.h",
        "bitstring.h",
        "channel.h",
        "channels_cirq.h",
        "circuit_noisy.h",
        "circuit_qsim_parser.h",
        "circuit.h",
        "expect.h",
        "formux.h",
        "fuser.h",
        "fuser_basic.h",
        "fuser_mqubit.h",
        "gate.h",
        "gate_appl.h",
        "gates_cirq.h",
        "gates_qsim.h",
        "hybrid.h",
        "io_file.h",
        "io.h",
        "matrix.h",
        "parfor.h",
        "qtrajectory.h",
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
        "umux.h",
        "unitaryspace.h",
        "unitaryspace_avx.h",
        "unitaryspace_basic.h",
        "unitaryspace_sse.h",
        "unitary_calculator_avx.h",
        "unitary_calculator_basic.h",
        "unitary_calculator_sse.h",
        "util.h",
        "vectorspace.h",
    ],
)

# Library to run qsim with qsim circuit parser and parallel `for`
cc_library(
    name = "run_qsim_lib",
    hdrs = [
        "bits.h",
        "circuit.h",
        "circuit_qsim_parser.h",
        "expect.h",
        "formux.h",
        "fuser.h",
        "fuser_basic.h",
        "fuser_mqubit.h",
        "gate.h",
        "gate_appl.h",
        "gates_qsim.h",
        "io.h",
        "io_file.h",
        "matrix.h",
        "parfor.h",
        "run_qsim.h",
        "seqfor.h",
        "simmux.h",
        "simulator_avx.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace.h",
        "statespace_avx.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "umux.h",
        "unitaryspace.h",
        "unitaryspace_avx.h",
        "unitaryspace_basic.h",
        "unitaryspace_sse.h",
        "unitary_calculator_avx.h",
        "unitary_calculator_basic.h",
        "unitary_calculator_sse.h",
        "util.h",
        "vectorspace.h",
    ],
)

# Library to run qsimh with qsim circuit parser and parallel `for`
cc_library(
    name = "run_qsimh_lib",
    hdrs = [
        "bits.h",
        "circuit.h",
        "circuit_qsim_parser.h",
        "expect.h",
        "formux.h",
        "fuser.h",
        "fuser_basic.h",
        "fuser_mqubit.h",
        "gate.h",
        "gate_appl.h",
        "gates_qsim.h",
        "hybrid.h",
        "io.h",
        "io_file.h",
        "matrix.h",
        "parfor.h",
        "run_qsimh.h",
        "seqfor.h",
        "simmux.h",
        "simulator_avx.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace.h",
        "statespace_avx.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "util.h",
        "vectorspace.h",
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
    name = "bits",
    hdrs = ["bits.h"],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deps = [":bits"],
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

# Both parallelism control paths with multiplexer
cc_library(
    name = "formux",
    hdrs = ["formux.h"],
    deps = [
        ":parfor",
        ":seqfor",
    ],
)

### Gate libraries ###

cc_library(
    name = "gate",
    hdrs = ["gate.h"],
    deps = [":matrix"],
)

cc_library(
    name = "gate_appl",
    hdrs = ["gate_appl.h"],
    deps = [
        ":fuser",
        ":gate",
        ":matrix",
    ],
)

cc_library(
    name = "gates_cirq",
    hdrs = ["gates_cirq.h"],
    deps = [
        ":gate",
        ":matrix",
    ],
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
    deps = [
        ":gate",
        ":matrix",
    ],
)

cc_library(
    name = "fuser_basic",
    hdrs = ["fuser_basic.h"],
    deps = [
        ":fuser",
        ":gate",
    ],
)

cc_library(
    name = "fuser_mqubit",
    hdrs = ["fuser_mqubit.h"],
    deps = [
        ":fuser",
        ":gate",
    ],
)

### Expectation value library ###

cc_library(
    name = "expect",
    hdrs = ["expect.h"],
    deps = [
        ":fuser",
        ":gate_appl",
    ],
)

### Helper libraries to run qsim and qsimh ###

cc_library(
    name = "run_qsim",
    hdrs = ["run_qsim.h"],
    deps = [
        ":gate",
        ":gate_appl",
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

### Vectorspace library ###

cc_library(
    name = "vectorspace",
    hdrs = ["vectorspace.h"],
)

### Statespace libraries ###

cc_library(
    name = "statespace",
    hdrs = ["statespace.h"],
    deps = [
        ":util",
        ":vectorspace",
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

### Simulator libraries ###

cc_library(
    name = "simulator_avx",
    hdrs = ["simulator_avx.h"],
    deps = [
        ":bits",
        ":statespace_avx",
    ],
)

cc_library(
    name = "simulator_basic",
    hdrs = ["simulator_basic.h"],
    deps = [
        ":bits",
        ":statespace_basic",
    ],
)

cc_library(
    name = "simulator_sse",
    hdrs = ["simulator_sse.h"],
    deps = [
        ":bits",
        ":statespace_sse",
    ],
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
        ":gate_appl",
    ],
)

### Channel and noisy circuit libraries ###

cc_library(
    name = "channel",
    hdrs = ["channel.h"],
    deps = [":gate"],
)

cc_library(
    name = "circuit_noisy",
    hdrs = ["circuit_noisy.h"],
    deps = [
        ":channel",
        ":circuit",
    ],
)

cc_library(
    name = "channels_cirq",
    hdrs = ["channels_cirq.h"],
    deps = [
        ":channel",
        ":gates_cirq",
    ],
)

### Quantum trajectory simulator ###

cc_library(
    name = "qtrajectory",
    hdrs = ["qtrajectory.h"],
    deps = [
        ":circuit_noisy",
        ":gate",
        ":gate_appl",
    ],
)

### UnitarySpace libraries ###

cc_library(
    name = "unitaryspace",
    hdrs = ["unitaryspace.h"],
    deps = [":vectorspace"],
)

cc_library(
    name = "unitaryspace_avx",
    hdrs = ["unitaryspace_avx.h"],
    deps = [":unitaryspace"],
)

cc_library(
    name = "unitaryspace_basic",
    hdrs = ["unitaryspace_basic.h"],
    deps = [":unitaryspace"],
)

cc_library(
    name = "unitaryspace_sse",
    hdrs = ["unitaryspace_sse.h"],
    deps = [":unitaryspace"],
)

### Unitary calculator libraries ###

cc_library(
    name = "unitary_calculator_avx",
    hdrs = ["unitary_calculator_avx.h"],
    deps = [
        ":bits",
        ":unitaryspace_avx"
    ],
)

cc_library(
    name = "unitary_calculator_basic",
    hdrs = ["unitary_calculator_basic.h"],
    deps = [
        ":bits",
        ":unitaryspace_basic"
    ],
)

cc_library(
    name = "unitary_calculator_sse",
    hdrs = ["unitary_calculator_sse.h"],
    deps = [
        ":bits",
        ":unitaryspace_sse"
    ],
)

### Unitary mux header ###

cc_library(
    name = "umux",
    hdrs = ["umux.h"],
    deps = [
        ":unitary_calculator_avx",
        ":unitary_calculator_basic",
        ":unitary_calculator_sse",
    ],
)
