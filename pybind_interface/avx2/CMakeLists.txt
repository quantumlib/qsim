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
    set(CMAKE_CXX_FLAGS "/arch:AVX2 /O2 /openmp")
ELSE()
    set(CMAKE_CXX_FLAGS "-mavx2 -mfma -O3")
ENDIF()

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

INCLUDE(../GetPybind11.cmake)
pybind11_add_module(qsim_avx2 pybind_main_avx2.cpp)

target_link_libraries(qsim_avx2 PUBLIC OpenMP::OpenMP_CXX)
