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

"""Unit tests for compiler_probe.bzl."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//dev_tools:compiler_probe.bzl", "get_compiler_flags", "get_cpu_features", "get_feature_booleans")

def _test_linux_basic_impl(ctx):
    env = unittest.begin(ctx)

    # Mock /proc/cpuinfo output for an older AVX/SSE2 Linux box
    cmd_output = "flags : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 cx16 pcid movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust erms invpcid rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves arat umip arch_capabilities"

    features = get_cpu_features("linux", cmd_output)
    asserts.true(env, features["avx"])
    asserts.false(env, features["avx2"])
    asserts.true(env, features["sse2"])
    asserts.false(env, features["sse4"])

    avx_copts, sse_copts = get_compiler_flags("linux", features)
    asserts.equals(env, ["-mavx"], avx_copts)
    asserts.equals(env, ["-msse2"], sse_copts)

    has_avx, has_sse, features_str = get_feature_booleans(features)
    asserts.true(env, has_avx)
    asserts.true(env, has_sse)
    asserts.true(env, "AVX" in features_str)
    asserts.true(env, "SSE2" in features_str)

    return unittest.end(env)

def _test_linux_avx2_impl(ctx):
    env = unittest.begin(ctx)

    # Mock /proc/cpuinfo output for a standard AVX2 Linux box
    cmd_output = "flags : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves arat umip arch_capabilities"

    features = get_cpu_features("linux", cmd_output)
    asserts.true(env, features["avx2"])
    asserts.true(env, features["fma"])
    asserts.true(env, features["bmi2"])
    asserts.true(env, features["sse4"])
    asserts.false(env, features["avx512f"])

    avx_copts, sse_copts = get_compiler_flags("linux", features)
    asserts.equals(env, ["-mavx2", "-mfma", "-mbmi2"], avx_copts)
    asserts.equals(env, ["-msse4"], sse_copts)

    return unittest.end(env)

def _test_linux_avx512_impl(ctx):
    env = unittest.begin(ctx)

    # Mock /proc/cpuinfo output for an AVX512 Linux box
    cmd_output = "flags : ... avx512f avx512dq avx512cd avx512bw avx512vl avx2 bmi2 ..."

    features = get_cpu_features("linux", cmd_output)
    asserts.true(env, features["avx512f"])
    asserts.true(env, features["bmi2"])

    avx_copts, _ = get_compiler_flags("linux", features)
    asserts.equals(env, ["-mavx512f", "-mbmi2"], avx_copts)

    return unittest.end(env)

def _test_macos_impl(ctx):
    env = unittest.begin(ctx)

    # Mock sysctl -a output for macOS
    cmd_output = "machdep.cpu.features: FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX EST TM2 SSSE3 FMA CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES PCID XSAVE OSXSAVE SEGLIM64 VMM TSCTMR AVX1.0 RDRAND F16C\nmachdep.cpu.leaf7_features: SMEP ERMS RDWRFSGS TSC_THREAD_OFFSET BMI1 HLE AVX2 BMI2 INVPCID RTM RDSEED ADX SMAP CLFLUSHOPT IPT"

    features = get_cpu_features("darwin", cmd_output)
    asserts.true(env, features["avx2"])
    asserts.true(env, features["sse4"])

    avx_copts, sse_copts = get_compiler_flags("darwin", features)
    asserts.equals(env, ["-mavx2", "-mfma"], avx_copts)
    asserts.equals(env, ["-msse4"], sse_copts)

    return unittest.end(env)

def _test_windows_impl(ctx):
    env = unittest.begin(ctx)

    # Windows logic is currently a heuristic
    features = get_cpu_features("windows", "")
    asserts.true(env, features["avx2"])

    avx_copts, _ = get_compiler_flags("windows", features)
    asserts.equals(env, ["/arch:AVX2"], avx_copts)

    return unittest.end(env)

linux_basic_test = unittest.make(_test_linux_basic_impl)
linux_avx2_test = unittest.make(_test_linux_avx2_impl)
linux_avx512_test = unittest.make(_test_linux_avx512_impl)
macos_test = unittest.make(_test_macos_impl)
windows_test = unittest.make(_test_windows_impl)

def compiler_probe_test_suite(name):
    unittest.suite(
        name,
        linux_basic_test,
        linux_avx2_test,
        linux_avx512_test,
        macos_test,
        windows_test,
    )
