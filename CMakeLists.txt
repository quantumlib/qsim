# Copyright 2019 Google LLC. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.31)
project(qsim LANGUAGES CXX)

# This text is prepended to messages printed by this config file so it's
# easier to figure out what came from where in the logs.
set(MSG_PREFIX "[qsim cmake configuration]")


# ~~~~~ Analyze the host's hardware & software features ~~~~~

include(CheckLanguage)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

check_language(CUDA)

# CMake normally sets CMAKE_APPLE_SILICON_PROCESSOR on Apple Silicon; however,
# it doesn't happen when running builds using cibuildwheel, even on Apple
# Silicon. We have had better luck checking and seting it ourselves.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(CMAKE_APPLE_SILICON_PROCESSOR TRUE)
    message(STATUS "${MSG_PREFIX} detected Apple Silicon")
else()
    set(CMAKE_APPLE_SILICON_PROCESSOR FALSE)
    message(STATUS "${MSG_PREFIX} did not detect Apple Silicon")
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "${MSG_PREFIX} found CUDA compiler "
                   "${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    message(STATUS "${MSG_PREFIX} did not find CUDA compiler")
    # Did not find the CUDA framewwork, so check for the HIP as an alternative.
    execute_process(COMMAND which hipcc
                    OUTPUT_VARIABLE has_hipcc
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(has_hipcc)
        message(STATUS "${MSG_PREFIX} found hipcc")
        project(qsim LANGUAGES CXX HIP)
    else()
        message(STATUS "${MSG_PREFIX} did not find hipcc")
    endif()
endif()

find_package(OpenMP REQUIRED)

cmake_host_system_information(RESULT HAVE_SSE2 QUERY HAS_SSE2)

if(WIN32)
    check_cxx_compiler_flag("/arch:AVX2" HAVE_AVX2)
    check_cxx_compiler_flag("/arch:AVX512" HAVE_AVX512)
else()
    check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
    check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
endif()


# ~~~~~ Configure the build ~~~~~

# The following settings mirror what is in our hand-written Makefiles.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options propagated to all sub-cmakefiles.
if(WIN32)
    add_compile_options(/O2 /std:c++17 /openmp)
else()
    add_compile_options(-O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -flto=auto)
endif()

# Always build the basic part.
add_subdirectory(pybind_interface/basic)
add_subdirectory(pybind_interface/decide)

# Add subdirectories based on the architecture or available compilers.
if(NOT CMAKE_APPLE_SILICON_PROCESSOR)
    if(CMAKE_CUDA_COMPILER)
        add_subdirectory(pybind_interface/cuda)
        if(DEFINED ENV{CUQUANTUM_ROOT})
            add_subdirectory(pybind_interface/custatevec)
        endif()
    elseif(has_hipcc)
        add_subdirectory(pybind_interface/hip)
    endif()

    if(HAVE_SSE2)
      add_subdirectory(pybind_interface/sse)
    endif()

    if(HAVE_AVX2)
      add_subdirectory(pybind_interface/avx2)
    endif()

    if(HAVE_AVX512)
      add_subdirectory(pybind_interface/avx512)
    endif()
endif()

if(APPLE)
    include_directories(
      "/usr/local/include"
      "/usr/local/opt/llvm/include"
      "/opt/homebrew/include"
      "/opt/homebrew/opt/llvm@19/include"
    )
    link_directories(
      "/usr/local/lib"
      "/usr/local/opt/llvm/lib"
      "/opt/homebrew/lib"
      "/opt/homebrew/opt/llvm@19/lib"
    )
endif()


# ~~~~~ Print misc. info ~~~~~

message(STATUS "${MSG_PREFIX} host has SSE2 = ${HAVE_SSE2}")
message(STATUS "${MSG_PREFIX} host has AVX2 = ${HAVE_AVX2}")
message(STATUS "${MSG_PREFIX} host has AVX512 = ${HAVE_AVX512}")

message(STATUS "${MSG_PREFIX} shell $PATH = $ENV{PATH}")
message(STATUS "${MSG_PREFIX} shell $CUQUANTUM_ROOT = $ENV{CUQUANTUM_ROOT}")
