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

cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:run_qsim_lib",
    ],
    data = ["//circuits:circuit_q24"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:bitstring",
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    copts = select({
        ":windows": windows_copts,
        "//conditions:default": [],
    }),
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)
