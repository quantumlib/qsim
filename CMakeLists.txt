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

# ~~~~~ Set project-wide policies ~~~~~

# The following settings mirror what is in our hand-written Makefiles.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For deduplication of static libraries, pick the first occurrence.
cmake_policy(SET CMP0179 NEW)

# ~~~~~ Analyze the host's hardware & software features ~~~~~

find_package(OpenMP REQUIRED COMPONENTS CXX NO_POLICY_SCOPE)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP CXX support not found")
endif()

# CMake normally sets CMAKE_APPLE_SILICON_PROCESSOR on Apple Silicon; however,
# it doesn't happen when running builds using cibuildwheel, even on Apple
# Silicon. It's more reliable to check and set it ourselves.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(CMAKE_APPLE_SILICON_PROCESSOR TRUE)
    message(VERBOSE "Detected Apple Silicon")
else()
    set(CMAKE_APPLE_SILICON_PROCESSOR FALSE)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(VERBOSE "Have ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    # If CUDA is not available, check for HIP as an alternative.
    message(VERBOSE "Did not find CUDA compiler; cannot support CUDA.")
    find_program(_HIP_COMPILER hipcc)
    if(_HIP_COMPILER)
        enable_language(HIP)
        message(VERBOSE "Found hipcc. Enabling HIP support.")
    else()
        message(VERBOSE "Did not find hipcc; cannot support HIP.")
    endif()
endif()

include(dev_tools/cmake/CheckCPU.cmake)
# Note: CMake uses "WIN32" for Windows targets, including Win64.
if(WIN32)
    check_cpu_support("avx2" CPU_SUPPORTS_AVX2)
    check_cpu_support("avx512f" CPU_SUPPORTS_AVX512)
    check_cpu_support("sse4.1" CPU_SUPPORTS_SSE4)
elseif(LINUX)
    check_cpu_support("avx2" CPU_SUPPORTS_AVX2)
    check_cpu_support("avx512f" CPU_SUPPORTS_AVX512)
    check_cpu_support("sse4" CPU_SUPPORTS_SSE4)
elseif(APPLE AND NOT CMAKE_APPLE_SILICON_PROCESSOR)
    check_cpu_support("avx2_0" CPU_SUPPORTS_AVX2)
    check_cpu_support("avx512f" CPU_SUPPORTS_AVX512)
    check_cpu_support("sse4_1" CPU_SUPPORTS_SSE4)
endif()

# ~~~~~ Configure the build ~~~~~

if(WIN32)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -D_GLIBCXX_USE_CXX11_ABI=1)
endif()

find_package(Python3 COMPONENTS Interpreter Development)

include(dev_tools/cmake/GetPybind11.cmake)

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
    elseif(HAVE_HIPCC)
        add_subdirectory(pybind_interface/hip)
    endif()

    if(CPU_SUPPORTS_AVX2)
      add_subdirectory(pybind_interface/avx2)
    endif()

    if(CPU_SUPPORTS_AVX512)
      add_subdirectory(pybind_interface/avx512)
    endif()

    if(CPU_SUPPORTS_SSE4)
      add_subdirectory(pybind_interface/sse)
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

# ~~~~~ Print info for debugging ~~~~~

message(DEBUG "Shell $PATH = $ENV{PATH}")
message(DEBUG "Shell $CUQUANTUM_ROOT = $ENV{CUQUANTUM_ROOT}")
