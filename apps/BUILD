# TODO: remove reliance on getopt (unistd.h) to allow apps to run on Windows.

cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    deps = [
        "//lib:run_qsim_lib",
    ],
    data = ["//circuits:circuit_q24"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    deps = [
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)
