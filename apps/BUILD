cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io_file",
        "//lib:parfor",
        "//lib:run_qsim",
        "//lib:simulator",
    ],
    data = ["//circuits:circuit_q24"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    deps = [
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io_file",
        "//lib:parfor",
        "//lib:run_qsim",
        "//lib:simulator",
    ],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io_file",
        "//lib:parfor",
        "//lib:run_qsim",
        "//lib:simulator",
        "//lib:util",
    ],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io_file",
        "//lib:parfor",
        "//lib:run_qsimh",
        "//lib:simulator",
        "//lib:util",
    ],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:circuit_qsim_parser",
        "//lib:fuser_basic",
        "//lib:gates_qsim",
        "//lib:io_file",
        "//lib:parfor",
        "//lib:run_qsimh",
        "//lib:simulator",
        "//lib:util",
    ],
)
