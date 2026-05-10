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

load("@local_compiler_config//:compiler_config.bzl", "HOST_HAS_AVX", "HOST_HAS_SSE")

def qsim_select_copts(native_copts, avx_copts, sse_copts, windows_copts, default = []):
    """Returns a select block for qsim compiler options based on active config.

    Handles AVX+SSE combination and errors on invalid mixtures with native.

    Args:
        native_copts: Compiler options to use when --config=native is active.
        avx_copts: Compiler options to use when AVX is requested.
        sse_copts: Compiler options to use when SSE is requested.
        windows_copts: Compiler options to use when building on Windows.
        default: Fallback compiler options if no specific configuration is active.

    Returns:
        A select block containing the appropriate compiler flags.
    """
    return select({
        "//:native_requested": native_copts,
        "//:avx_and_sse_requested": avx_copts + sse_copts,
        "//:avx_requested": avx_copts,
        "//:sse_requested": sse_copts,
        "@platforms//os:windows": windows_copts,
        "//conditions:default": default,
    })

def qsim_feature_compatibility(feature):
    """Returns a select block for target_compatible_with based on required feature.

    Ensures SIMD tests are only built/run when the corresponding config is active
    OR when the host supports it in native mode.

    Args:
        feature: The feature name to check for ("avx" or "sse").

    Returns:
        A select block that evaluates to an empty list if compatible, or
        ["@platforms//:incompatible"] otherwise.
    """
    if feature == "avx":
        native_compatible = [] if HOST_HAS_AVX else ["@platforms//:incompatible"]
        return select({
            "//:avx_requested": [],
            "//:avx_and_sse_requested": [],
            "//:native_requested": native_compatible,
            "//conditions:default": ["@platforms//:incompatible"],
        })
    elif feature == "sse":
        native_compatible = [] if HOST_HAS_SSE else ["@platforms//:incompatible"]
        return select({
            "//:sse_requested": [],
            "//:avx_and_sse_requested": [],
            "//:native_requested": native_compatible,
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
