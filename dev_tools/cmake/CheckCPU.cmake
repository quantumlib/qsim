# Copyright 2025 Google LLC
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

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

macro(check_cpu_support _FEATURE_STRING _FEATURE_FLAG)
    set(${_FEATURE_FLAG} FALSE)

    message(STATUS "Testing platform support for ${_FEATURE_STRING}")
    if(WIN32)
        # On Windows, there's no built-in method to learn the CPU flags. Third-
        # party tools exist, but downloading & running them is a security risk.
        # We resort instead to compiling and running our own small program.
        set(_CHECKER_FILE_PATH "${CMAKE_BINARY_DIR}/checker.cpp")
        file(WRITE ${_CHECKER_FILE_PATH} "${_WIN32_CHECKER_SRC}")
        try_run(
            _CHECKER_RETURN_VALUE
            _CHECKER_COMPILED
            "${CMAKE_BINARY_DIR}"
            "${_CHECKER_FILE_PATH}"
            RUN_OUTPUT_VARIABLE _CPU_FEATURES
        )
        if(_CHECKER_COMPILED AND _CHECKER_RETURN_VALUE EQUAL 0)
            string(FIND "${_CPU_FEATURES}" ${_FEATURE_STRING} _FOUND)
            if(NOT _FOUND EQUAL -1)
                set(${_FEATURE_FLAG} TRUE)
            endif()
        else()
            message(STATUS "Unable to autodetect vector instruction sets")
            if(NOT _CHECKER_COMPILED)
                message(STATUS " (failed to compile CPU checker utility)")
            else()
                message(STATUS " (got an error trying to run our CPU checker)")
            endif()
        endif()

    elseif(LINUX)
        execute_process(
            COMMAND bash --noprofile -c "grep -q ${_FEATURE_STRING} /proc/cpuinfo"
            RESULT_VARIABLE _EXIT_CODE
        )
        if(_EXIT_CODE EQUAL 0)
            set(${_FEATURE_FLAG} TRUE)
        endif()

    elseif(APPLE AND NOT CMAKE_APPLE_SILICON_PROCESSOR)
        execute_process(
            COMMAND bash --noprofile -c "sysctl -n hw.optional.${_FEATURE_STRING}"
            RESULT_VARIABLE _EXIT_CODE
            OUTPUT_VARIABLE _FLAG_VALUE
        )
        if(_EXIT_CODE EQUAL 0 AND _FLAG_VALUE EQUAL "1")
            set(${_FEATURE_FLAG} TRUE)
        endif()
    endif()

    if(${_FEATURE_FLAG})
        message(STATUS "Testing platform support for ${_FEATURE_STRING} - found")
    else()
        message(STATUS "Testing platform support for ${_FEATURE_STRING} - not found")
    endif()
endmacro()

# Small Windows C++ program to test bits in certain Intel CPU registers.
# Info about the registers in Intel CPUs: https://en.wikipedia.org/wiki/CPUID
#
# EAX  ECX  Bit   Name
#  1    0   19    sse4.1
#  1    0   20    sse4.2
#  1    0   28    avx
#  7    0    5    avx2
#  7    0   16    avx512f
#
# Note: CMake caches the output of try_run() by default; therefore, this program
# will not be executed each time try_run() is called.

set(_WIN32_CHECKER_SRC "
#include <iostream>
#include <string>
#include <intrin.h>

int main() {
    int cpuInfo[4];
    __cpuidex(cpuInfo, 1, 0);
    std::cout << ((cpuInfo[2] & (1 << 19)) ? \"sse4.1\\n\"  : \"\");
    std::cout << ((cpuInfo[2] & (1 << 20)) ? \"sse4.2\\n\"  : \"\");
    __cpuidex(cpuInfo, 7, 0);
    std::cout << ((cpuInfo[1] & (1 << 5))  ? \"avx2\\n\"    : \"\")
              << ((cpuInfo[1] & (1 << 16)) ? \"avx512f\\n\" : \"\");
    return 0;
}
")
