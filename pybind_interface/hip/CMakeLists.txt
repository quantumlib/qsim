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
project(qsim LANGUAGES CXX HIP)

if(WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /openmp")
else()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

INCLUDE(../GetPybind11.cmake)
find_package(PythonLibs 3.10 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")
find_package(HIP REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)

hip_add_library(qsim_hip MODULE pybind_main_hip.cpp)

set_target_properties(qsim_hip PROPERTIES
       PREFIX "${PYTHON_MODULE_PREFIX}"
       SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_hip.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(qsim_hip PUBLIC OpenMP::OpenMP_CXX)
