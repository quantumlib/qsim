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

set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.31)
project(qsim LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_APPLE_SILICON_PROCESSOR)
    # When running builds using cibuildwheel, the value is not set even when
    # running on Apple Silicon. Check and set it ourselves if we have to.
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(CMAKE_APPLE_SILICON_PROCESSOR TRUE)
        message(STATUS "qsim: detected Apple Silicon")
    else()
        set(CMAKE_APPLE_SILICON_PROCESSOR FALSE)
        message(STATUS "qsim: did not detect Apple Silicon")
    endif()
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "qsim: found CUDA compiler ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    message(STATUS "qsim: did not find CUDA compiler")
    execute_process(COMMAND which hipcc OUTPUT_VARIABLE has_hipcc OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(has_hipcc)
        message(STATUS "qsim: found hipcc")
        project(qsim LANGUAGES CXX HIP)
    else()
        message(STATUS "qsim: did not find nvcc or hipcc")
    endif()
endif()

find_package(OpenMP REQUIRED)

# Always build the basic part.
ADD_SUBDIRECTORY(pybind_interface/basic)
ADD_SUBDIRECTORY(pybind_interface/decide)

# Add subdirectories based on the architecture or available compilers.
if(NOT CMAKE_APPLE_SILICON_PROCESSOR)
    if(CMAKE_CUDA_COMPILER)
        ADD_SUBDIRECTORY(pybind_interface/cuda)
        if(DEFINED ENV{CUQUANTUM_ROOT})
            ADD_SUBDIRECTORY(pybind_interface/custatevec)
        endif()
    elseif(has_hipcc)
        ADD_SUBDIRECTORY(pybind_interface/hip)
    endif()

    ADD_SUBDIRECTORY(pybind_interface/sse)
    ADD_SUBDIRECTORY(pybind_interface/avx512)
    ADD_SUBDIRECTORY(pybind_interface/avx2)
endif()
