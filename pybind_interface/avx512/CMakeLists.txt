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
project(qsim)

IF (WIN32)
    add_compile_options(/arch:AVX512)
ELSEIF(APPLE AND NOT CMAKE_APPLE_SILICON_PROCESSOR
       AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # On Intel Macs CMake detects AVX but clang doesn't recognize -mavx512.
    # Using arch=native should turn on the features if they're available.
    add_compile_options(-march=native)
ELSE()
    add_compile_options(-mavx512f -mbmi2)
ENDIF()

INCLUDE(../GetPybind11.cmake)
pybind11_add_module(qsim_avx512 pybind_main_avx512.cpp)

target_link_libraries(qsim_avx512 PUBLIC OpenMP::OpenMP_CXX)
