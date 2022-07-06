package(default_visibility = ["//visibility:public"])

# Libraries of the following form:
#   # cuda_library
#   cc_library(...)
# are converted to cuda_library rules when imported to the Google codebase.
# Do not modify this tag.

##### Aggregate libraries #####

# Full qsim library, minus CUDA
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
        "mps_simulator.h",
        "mps_statespace.h",
        "parfor.h",
        "qtrajectory.h",
        "run_qsim.h",
        "run_qsimh.h",
        "seqfor.h",
        "simmux.h",
        "simulator.h",
        "simulator_avx.h",
        "simulator_avx512.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace_avx.h",
        "statespace_avx512.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "statespace.h",
        "umux.h",
        "unitaryspace.h",
        "unitaryspace_avx.h",
        "unitaryspace_avx512.h",
        "unitaryspace_basic.h",
        "unitaryspace_sse.h",
        "unitary_calculator_avx.h",
        "unitary_calculator_avx512.h",
        "unitary_calculator_basic.h",
        "unitary_calculator_sse.h",
        "util.h",
        "util_cpu.h",
        "vectorspace.h",
    ],
)

# Full qsim library, including CUDA
# cuda_library
cc_library(
    name = "qsim_cuda_lib",
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
        "mps_simulator.h",
        "mps_statespace.h",
        "parfor.h",
        "qtrajectory.h",
        "run_qsim.h",
        "run_qsimh.h",
        "seqfor.h",
        "simmux.h",
        "simulator.h",
        "simulator_avx.h",
        "simulator_avx512.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "simulator_cuda.h",
        "simulator_cuda_kernels.h",
        "statespace_avx.h",
        "statespace_avx512.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "statespace_cuda.h",
        "statespace_cuda_kernels.h",
        "statespace.h",
        "umux.h",
        "unitaryspace.h",
        "unitaryspace_avx.h",
        "unitaryspace_avx512.h",
        "unitaryspace_basic.h",
        "unitaryspace_sse.h",
        "unitary_calculator_avx.h",
        "unitary_calculator_avx512.h",
        "unitary_calculator_basic.h",
        "unitary_calculator_sse.h",
        "util.h",
        "util_cpu.h",
        "util_cuda.h",
        "vectorspace.h",
        "vectorspace_cuda.h",
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
        "simulator.h",
        "simulator_avx.h",
        "simulator_avx512.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace.h",
        "statespace_avx.h",
        "statespace_avx512.h",
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
        "util_cpu.h",
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
        "simulator.h",
        "simulator_avx.h",
        "simulator_avx512.h",
        "simulator_basic.h",
        "simulator_sse.h",
        "statespace.h",
        "statespace_avx.h",
        "statespace_avx512.h",
        "statespace_basic.h",
        "statespace_sse.h",
        "util.h",
        "util_cpu.h",
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

cc_library(
    name = "util_cpu",
    hdrs = ["util_cpu.h"],
)

# cuda_library
cc_library(
    name = "util_cuda",
    hdrs = ["util_cuda.h"],
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

### Vectorspace libraries ###

cc_library(
    name = "vectorspace",
    hdrs = ["vectorspace.h"],
)

# cuda_library
cc_library(
    name = "vectorspace_cuda",
    hdrs = ["vectorspace_cuda.h"],
)

### Statespace libraries ###

cc_library(
    name = "statespace",
    hdrs = ["statespace.h"],
    deps = [":util"],
)

cc_library(
    name = "statespace_avx",
    hdrs = ["statespace_avx.h"],
    deps = [
        ":statespace",
        ":util",
        ":vectorspace",
    ],
)

cc_library(
    name = "statespace_avx512",
    hdrs = ["statespace_avx512.h"],
    deps = [
        ":statespace",
        ":util",
        ":vectorspace",
    ],
)

cc_library(
    name = "statespace_basic",
    hdrs = ["statespace_basic.h"],
    deps = [
        ":statespace",
        ":util",
        ":vectorspace",
    ],
)

cc_library(
    name = "statespace_sse",
    hdrs = ["statespace_sse.h"],
    deps = [
        ":statespace",
        ":util",
        ":vectorspace",
    ],
)

# cuda_library
cc_library(
    name = "statespace_cuda",
    hdrs = [
        "statespace_cuda.h",
        "statespace_cuda_kernels.h",
    ],
    deps = [
        ":statespace",
        ":util_cuda",
        ":vectorspace_cuda",
    ],
)

### Simulator libraries ###

cc_library(
    name = "simulator_base",
    hdrs = ["simulator.h"],
    deps = [":bits"],
)

cc_library(
    name = "simulator_avx",
    hdrs = ["simulator_avx.h"],
    deps = [
        ":simulator_base",
        ":statespace_avx",
    ],
)

cc_library(
    name = "simulator_avx512",
    hdrs = ["simulator_avx512.h"],
    deps = [
        ":simulator_base",
        ":statespace_avx512",
    ],
)

cc_library(
    name = "simulator_basic",
    hdrs = ["simulator_basic.h"],
    deps = [
        ":simulator_base",
        ":statespace_basic",
    ],
)

cc_library(
    name = "simulator_sse",
    hdrs = ["simulator_sse.h"],
    deps = [
        ":simulator_base",
        ":statespace_sse",
    ],
)

# cuda_library
cc_library(
    name = "simulator_cuda",
    hdrs = [
        "simulator_cuda.h",
        "simulator_cuda_kernels.h",
    ],
    deps = [
        ":bits",
        ":statespace_cuda",
        ":util_cuda",
    ],
)

# All three state-vector simulators with multiplexer
cc_library(
    name = "simulator",
    hdrs = ["simmux.h"],
    deps = [
        ":simulator_avx",
        ":simulator_avx512",
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
    deps = [
        ":gate",
        ":matrix",
    ],
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
)

cc_library(
    name = "unitaryspace_avx",
    hdrs = ["unitaryspace_avx.h"],
    deps = [
        ":unitaryspace",
        ":vectorspace",
    ],
)

cc_library(
    name = "unitaryspace_avx512",
    hdrs = ["unitaryspace_avx512.h"],
    deps = [
        ":unitaryspace",
        ":vectorspace",
    ],
)

cc_library(
    name = "unitaryspace_basic",
    hdrs = ["unitaryspace_basic.h"],
    deps = [
        ":unitaryspace",
        ":vectorspace",
    ],
)

cc_library(
    name = "unitaryspace_sse",
    hdrs = ["unitaryspace_sse.h"],
    deps = [
        ":unitaryspace",
        ":vectorspace",
    ],
)

### Unitary calculator libraries ###

cc_library(
    name = "unitary_calculator_avx",
    hdrs = ["unitary_calculator_avx.h"],
    deps = [
        ":simulator_base",
        ":unitaryspace_avx",
    ],
)

cc_library(
    name = "unitary_calculator_avx512",
    hdrs = ["unitary_calculator_avx512.h"],
    deps = [
        ":simulator_base",
        ":unitaryspace_avx512",
    ],
)

cc_library(
    name = "unitary_calculator_basic",
    hdrs = ["unitary_calculator_basic.h"],
    deps = [
        ":simulator_base",
        ":unitaryspace_basic",
    ],
)

cc_library(
    name = "unitary_calculator_sse",
    hdrs = ["unitary_calculator_sse.h"],
    deps = [
        ":simulator_base",
        ":unitaryspace_sse",
    ],
)

### Unitary mux header ###

cc_library(
    name = "umux",
    hdrs = ["umux.h"],
    deps = [
        ":unitary_calculator_avx",
        ":unitary_calculator_avx512",
        ":unitary_calculator_basic",
        ":unitary_calculator_sse",
    ],
)

### All MPS headers ###

cc_library(
    name = "mps_statespace",
    hdrs = ["mps_statespace.h"],
    deps = ["@eigen//:eigen3"],
)

cc_library(
    name = "mps_simulator",
    hdrs = ["mps_simulator.h"],
    deps = [
        ":mps_statespace",
        "@eigen//:eigen3",
    ],
)
