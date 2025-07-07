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

if(CMAKE_CUDA_ARCHITECTURES)
    # CMake 3.18+ sets this variable from $CUDAARCHS automatically.
    message(STATUS "using CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "using 'native' for the CUDA architecture value")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(qsim_cuda MODULE pybind_main_cuda.cpp)
set_target_properties(qsim_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_cuda.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(qsim_cuda PUBLIC OpenMP::OpenMP_CXX)
