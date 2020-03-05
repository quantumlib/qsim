cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    deps = ["//lib:qsim_lib"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    deps = ["//lib:qsim_lib"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    deps = ["//lib:qsim_lib"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    deps = ["//lib:qsim_lib"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    deps = ["//lib:qsim_lib"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)