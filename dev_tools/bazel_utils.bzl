# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for qsim Bazel builds."""

load("@local_compiler_config//:compiler_config.bzl", "AVX2_COPTS", "AVX512_COPTS", "AVX_COPTS", "BMI2_COPTS", "HOST_HAS_AVX", "HOST_HAS_AVX2", "HOST_HAS_AVX512", "HOST_HAS_SSE", "SSE_COPTS", "SUPPORTS_GSFRAME")

def qsim_select_copts(target_level = "basic"):
    """Returns a select block for qsim compiler options based on active config.

    Handles AVX variants and SSE combinations.

    Args:
        target_level: The instruction set level for this target.
            Options: "basic", "avx", "avx2", "avx512", "sse".
            If "basic", SIMD flags are only added if --config=native is active.

    Returns:
        A select block containing the appropriate compiler flags.
    """

    gsframe_flags = ["-Wa,--gsframe=no"] if SUPPORTS_GSFRAME else []
    linux_gsframe = select({
        "@platforms//os:linux": gsframe_flags,
        "//conditions:default": [],
    })

    # 1. Native configuration: always uses host optimization.
    native_part = select({
        "//:native_requested": ["-march=native"],
        "//conditions:default": [],
    })

    # 2. Main instruction set selection.
    # We flatten the logic into a single select per target_level.

    if target_level == "basic":
        main_select = select({
            "@platforms//os:windows": [],
            "//conditions:default": [],
        })
    elif target_level == "sse":
        main_select = select({
            "@platforms//os:windows": [],
            "//:sse_requested": SSE_COPTS,
            "//:native_requested": [],
            "//conditions:default": SSE_COPTS,
        })
    elif target_level == "avx":
        main_select = select({
            "@platforms//os:windows": AVX_COPTS,
            "//:avx512_requested": AVX_COPTS,
            "//:avx2_requested": AVX_COPTS,
            "//:avx_requested": AVX_COPTS,
            "//:native_requested": [],
            "//conditions:default": AVX_COPTS,
        })
    elif target_level == "avx2":
        main_select = select({
            "@platforms//os:windows": AVX_COPTS,
            "//:avx512_requested": AVX2_COPTS,
            "//:avx2_requested": AVX2_COPTS,
            "//:avx_requested": AVX_COPTS,
            "//:native_requested": [],
            "//conditions:default": AVX2_COPTS,
        })
    elif target_level == "avx512":
        main_select = select({
            "@platforms//os:windows": AVX_COPTS,
            "//:avx512_requested": AVX512_COPTS,
            "//:avx2_requested": AVX2_COPTS,
            "//:avx_requested": AVX_COPTS,
            "//:native_requested": [],
            "//conditions:default": AVX512_COPTS,
        })
    else:
        fail("Invalid target_level: " + target_level)

    # 3. SSE is additive for AVX configurations.
    sse_requested_val = SSE_COPTS if target_level != "sse" else []
    sse_select = select({
        "@platforms//os:windows": [],
        "//:native_requested": [],
        "//:sse_requested": sse_requested_val,
        "//conditions:default": [],
    })

    # 4. BMI2 is additive to everything except Windows and Native.
    bmi2_select = select({
        "@platforms//os:windows": [],
        "//:native_requested": [],
        "//conditions:default": BMI2_COPTS,
    })

    return native_part + main_select + sse_select + bmi2_select + linux_gsframe

def qsim_feature_compatibility(feature):
    """Returns a select block for target_compatible_with based on required feature.

    Ensures SIMD tests are only built/run when BOTH the corresponding config
    is active AND the host supports it.

    Args:
        feature: The feature name to check for ("avx", "avx2", "avx512" or "sse").

    Returns:
        A select block that evaluates to an empty list if compatible, or
        ["@platforms//:incompatible"] otherwise.
    """

    avx_ok = [] if HOST_HAS_AVX or HOST_HAS_AVX2 or HOST_HAS_AVX512 else ["@platforms//:incompatible"]
    avx2_ok = [] if HOST_HAS_AVX2 or HOST_HAS_AVX512 else ["@platforms//:incompatible"]
    avx512_ok = [] if HOST_HAS_AVX512 else ["@platforms//:incompatible"]
    sse_ok = [] if HOST_HAS_SSE else ["@platforms//:incompatible"]

    if feature == "avx":
        return select({
            "//:avx_requested": avx_ok,
            "//:avx2_requested": avx2_ok,
            "//:avx512_requested": avx512_ok,
            "//:avx_and_avx2_requested": avx_ok,
            "//:avx_and_avx512_requested": avx_ok,
            "//:avx512_and_avx2_requested": avx2_ok,
            "//:avx_all_requested": avx_ok,
            "//:avx_and_sse_requested": avx_ok,
            "//:native_requested": avx_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "avx2":
        return select({
            "//:avx2_requested": avx2_ok,
            "//:avx512_requested": avx512_ok,
            "//:avx512_and_avx2_requested": avx2_ok,
            "//:native_requested": avx2_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "avx512":
        return select({
            "//:avx512_requested": avx512_ok,
            "//:avx512_and_avx2_requested": avx512_ok,
            "//:native_requested": avx512_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "sse":
        return select({
            "//:sse_requested": sse_ok,
            "//:avx_and_sse_requested": sse_ok,
            "//:native_requested": sse_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    return []

def _qsim_print_flags_impl(ctx):
    if ctx.attr.flags:
        print("Active qsim compiler flags: " + " ".join(ctx.attr.flags))  # buildifier: disable=print
    return [CcInfo()]

qsim_print_flags = rule(
    implementation = _qsim_print_flags_impl,
    attrs = {
        "flags": attr.string_list(),
    },
    provides = [CcInfo],
)
