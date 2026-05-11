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

load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "@local_compiler_config//:compiler_config.bzl",
    "AVX2_COPTS",
    "AVX512_COPTS",
    "AVX_COPTS",
    "BMI2_COPTS",
    "BMI_COPTS",
    "HOST_HAS_AVX",
    "HOST_HAS_AVX2",
    "HOST_HAS_AVX512",
    "HOST_HAS_BMI",
    "HOST_HAS_BMI2",
    "HOST_HAS_SSE",
    "SSE_COPTS",
    "SUPPORTS_GSFRAME",
)

def qsim_select_copts(target_level = "basic"):
    """Returns a select block for qsim compiler options based on active config.

    Handles additive combinations of AVX, SSE, and BMI.

    Args:
        target_level: The maximum instruction set level for this target.
            Options: "basic", "avx", "avx2", "avx512", "sse".
            If "basic", SIMD flags are only added if requested.

    Returns:
        A list of select blocks that sum to the appropriate compiler flags.
    """

    # 1. Native configuration.
    native_part = select({
        "//:native_requested": ["-march=native"],
        "//conditions:default": [],
    })

    # 2. AVX part (additive)
    # If native is requested, we don't add specific AVX flags to avoid conflict.
    avx_part = select({
        "//:native_requested": [],
        "//:avx512_requested": AVX512_COPTS if target_level == "avx512" else (
            AVX2_COPTS if target_level == "avx2" else AVX_COPTS
        ),
        "//:avx2_requested": AVX2_COPTS if target_level in [
            "avx2",
            "avx512",
        ] else AVX_COPTS,
        "//:avx_requested": AVX_COPTS,
        "//conditions:default": [],
    })

    # 3. SSE part (additive)
    sse_part = select({
        "//:native_requested": [],
        "//:sse_requested": SSE_COPTS,
        "//conditions:default": [],
    })

    # 4. BMI part (additive)
    bmi_part = select({
        "//:native_requested": [],
        "//:bmi2_requested": BMI2_COPTS,
        "//:bmi_requested": BMI_COPTS,
        "//conditions:default": [],
    })

    # 5. Linux specific flags (SFrame)
    gsframe_flags = ["-Wa,--gsframe=no"] if SUPPORTS_GSFRAME else []
    gsframe_part = select({
        "@platforms//os:linux": gsframe_flags,
        "//conditions:default": [],
    })

    return native_part + avx_part + sse_part + bmi_part + gsframe_part

def _qsim_validate_host_impl(ctx):
    # Check for mutual exclusivity of native and specific flags.
    if ctx.attr.native_requested and (
        ctx.attr.avx_requested or ctx.attr.avx2_requested or
        ctx.attr.avx512_requested or ctx.attr.sse_requested or
        ctx.attr.bmi_requested or ctx.attr.bmi2_requested
    ):
        fail("--config=native is incompatible with specific architecture flags.")

    # Check host support for requested features.
    if ctx.attr.avx_requested and not HOST_HAS_AVX:
        fail("Requested AVX support, but host CPU does not support it.")
    if ctx.attr.avx2_requested and not HOST_HAS_AVX2:
        fail("Requested AVX2 support, but host CPU does not support it.")
    if ctx.attr.avx512_requested and not HOST_HAS_AVX512:
        fail("Requested AVX512 support, but host CPU does not support it.")
    if ctx.attr.sse_requested and not HOST_HAS_SSE:
        fail("Requested SSE support, but host CPU does not support it.")
    if ctx.attr.bmi_requested and not HOST_HAS_BMI:
        fail("Requested BMI support, but host CPU does not support it.")
    if ctx.attr.bmi2_requested and not HOST_HAS_BMI2:
        fail("Requested BMI2 support, but host CPU does not support it.")

    return [CcInfo()]

qsim_validate_host_rule = rule(
    implementation = _qsim_validate_host_impl,
    attrs = {
        "avx_requested": attr.bool(),
        "avx2_requested": attr.bool(),
        "avx512_requested": attr.bool(),
        "bmi_requested": attr.bool(),
        "bmi2_requested": attr.bool(),
        "native_requested": attr.bool(),
        "sse_requested": attr.bool(),
    },
    provides = [CcInfo],
)

def qsim_validate_host(name, **kwargs):
    """Enforces host compatibility and flag exclusivity."""
    qsim_validate_host_rule(
        name = name,
        avx_requested = select({
            "//:avx_requested": True,
            "//conditions:default": False,
        }),
        avx2_requested = select({
            "//:avx2_requested": True,
            "//conditions:default": False,
        }),
        avx512_requested = select({
            "//:avx512_requested": True,
            "//conditions:default": False,
        }),
        bmi_requested = select({
            "//:bmi_requested": True,
            "//conditions:default": False,
        }),
        bmi2_requested = select({
            "//:bmi2_requested": True,
            "//conditions:default": False,
        }),
        native_requested = select({
            "//:native_requested": True,
            "//conditions:default": False,
        }),
        sse_requested = select({
            "//:sse_requested": True,
            "//conditions:default": False,
        }),
        **kwargs
    )

def _qsim_print_config_impl(ctx):
    if ctx.attr.verbose:
        print("--- qsim Build Configuration ---")  # buildifier: disable=print
        print("Active compiler flags: " + " ".join(ctx.attr.flags))  # buildifier: disable=print
        print("--------------------------------")  # buildifier: disable=print
    return [CcInfo()]

qsim_print_config_rule = rule(
    implementation = _qsim_print_config_impl,
    attrs = {
        "flags": attr.string_list(),
        "verbose": attr.bool(),
    },
    provides = [CcInfo],
)

def qsim_print_config(name, **kwargs):
    """Prints build configuration if --config=verbose is active."""
    qsim_print_config_rule(
        name = name,
        flags = qsim_select_copts(target_level = "avx512"),
        verbose = select({
            "//:verbose_requested": True,
            "//conditions:default": False,
        }),
        **kwargs
    )

def qsim_feature_compatibility(feature):
    """Returns a select block for target_compatible_with based on required feature.

    Args:
        feature: The feature name to check for ("avx", "avx2", "avx512" or "sse").

    Returns:
        A select block that evaluates to an empty list if compatible, or
        ["@platforms//:incompatible"] otherwise.
    """

    # We reuse the logic: if native or specific flag is requested, check host.
    # If NO flag is requested, it might still be incompatible if it's a feature test.

    avx_ok = [] if HOST_HAS_AVX or HOST_HAS_AVX2 or HOST_HAS_AVX512 else [
        "@platforms//:incompatible",
    ]
    avx2_ok = [] if HOST_HAS_AVX2 or HOST_HAS_AVX512 else [
        "@platforms//:incompatible",
    ]
    avx512_ok = [] if HOST_HAS_AVX512 else ["@platforms//:incompatible"]
    sse_ok = [] if HOST_HAS_SSE else ["@platforms//:incompatible"]

    if feature == "avx":
        return selects.with_or({
            (
                "//:avx_requested",
                "//:avx2_requested",
                "//:avx512_requested",
                "//:native_requested",
            ): avx_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "avx2":
        return selects.with_or({
            (
                "//:avx2_requested",
                "//:avx512_requested",
                "//:native_requested",
            ): avx2_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "avx512":
        return selects.with_or({
            (
                "//:avx512_requested",
                "//:native_requested",
            ): avx512_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "sse":
        return selects.with_or({
            (
                "//:sse_requested",
                "//:native_requested",
            ): sse_ok,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    return []
