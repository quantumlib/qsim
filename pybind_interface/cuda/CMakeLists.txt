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
project(qsim LANGUAGES CXX CUDA)

if(WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /openmp")
else()
    set(CMAKE_CXX_FLAGS "-O3")
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

include(../GetPybind11.cmake)
include(../GetCUDAARCHS.cmake)

find_package(Python 3.10 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
if(pybind11_FOUND)
    include_directories(${pybind11_INCLUDE_DIRS})
else()  # means pybind11 has been fetched in GetPybind11.cmake
    include_directories(${pybind11_SOURCE_DIR}/include)
endif()

add_library(qsim_cuda MODULE pybind_main_cuda.cpp)
set_target_properties(qsim_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_cuda.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(qsim_cuda OpenMP::OpenMP_CXX)
