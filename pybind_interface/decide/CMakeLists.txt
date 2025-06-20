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

include(CheckLanguage)
check_language(CUDA)

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

# Configure based on the detected platform
if(CMAKE_CUDA_COMPILER)
    include(../GetCUDAARCHS.cmake)
    add_library(qsim_decide MODULE decide.cpp)
    if(DEFINED ENV{CUQUANTUM_ROOT})
        target_compile_options(qsim_decide PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-D__CUSTATEVEC__>
        )
    endif()
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)
    set_target_properties(qsim_decide PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(qsim_decide OpenMP::OpenMP_CXX)
elseif(has_hipcc)
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")
    find_package(HIP REQUIRED)
    hip_add_library(qsim_decide MODULE decide.cpp)
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE HIP)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)
    set_target_properties(qsim_decide PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    target_link_libraries(qsim_decide PUBLIC OpenMP::OpenMP_CXX)
else()
    pybind11_add_module(qsim_decide decide.cpp)
    target_link_libraries(qsim_decide PUBLIC OpenMP::OpenMP_CXX)
endif()

