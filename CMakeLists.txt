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

cmake_minimum_required(VERSION 3.28)
project(qsim LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

# This text is prepended to messages printed by this config file so it's
# easier to figure out what came from where in the logs.
set(MSG_PREFIX "[qsim cmake configuration]")

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

# On Macs, Homebrew's dirs are the likely places to find LLVM & OpenMP.
if(APPLE)
    find_program(BREW brew)
    if(BREW)
        execute_process(
            COMMAND ${BREW} --prefix
            OUTPUT_VARIABLE BREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        include_directories("${BREW_PREFIX}/include")
        link_directories("${BREW_PREFIX}/lib")
        execute_process(
            COMMAND ${BREW} --prefix llvm
            OUTPUT_VARIABLE LLVM_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(NOT LLVM_PREFIX)
        find_program(LLVM_CONFIG llvm-config-19)
        if(NOT LLVM_CONFIG)
            find_program(LLVM_CONFIG llvm-config)
        endif()
        if (LLVM_CONFIG)
            execute_process(
                COMMAND ${LLVM_CONFIG} --prefix
                OUTPUT_VARIABLE LLVM_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
        endif()
    endif()
    if(LLVM_PREFIX)
        list(PREPEND CMAKE_PREFIX_PATH "${LLVM_PREFIX}")
    endif()

    find_package(LLVM CONFIG REQUIRED)
    include_directories(${LLVM_INCLUDE_DIRS})
    link_directories(${LLVM_LIBRARY_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

find_package(OpenMP REQUIRED)

# Always build the basic part.
add_subdirectory(pybind_interface/basic)
add_subdirectory(pybind_interface/decide)

# Add subdirectories based on the architecture or available compilers.
if(NOT CMAKE_APPLE_SILICON_PROCESSOR)
    if(CMAKE_CUDA_COMPILER)
        add_subdirectory(pybind_interface/cuda)
        if(DEFINED ENV{CUQUANTUM_ROOT})
            add_subdirectory(pybind_interface/custatevec)
            add_subdirectory(pybind_interface/custatevecex)
        endif()
    elseif(has_hipcc)
        add_subdirectory(pybind_interface/hip)
    endif()

    add_subdirectory(pybind_interface/sse)
    add_subdirectory(pybind_interface/avx512)
    add_subdirectory(pybind_interface/avx2)
endif()

# Additional miscellanous settings.
# The following settings mirror what is in our hand-written Makefiles.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Print additional useful info.
message(STATUS "${MSG_PREFIX} OpenMP found = ${OPENMP_FOUND}")
message(STATUS "${MSG_PREFIX} shell $PATH = $ENV{PATH}")
