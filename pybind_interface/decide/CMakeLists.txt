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

# Configure based on the detected platform
if(CMAKE_CUDA_COMPILER)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    add_library(qsim_decide MODULE decide.cpp)
    if(DEFINED ENV{CUQUANTUM_ROOT})
        target_compile_options(qsim_decide PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-D__CUSTATEVEC__>
        )
    endif()
    set_target_properties(qsim_decide PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE CUDA)
elseif(has_hipcc)
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")
    find_package(HIP REQUIRED)
    hip_add_library(qsim_decide MODULE decide.cpp)
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(qsim_decide PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
else()
    pybind11_add_module(qsim_decide decide.cpp)
endif()

target_link_libraries(qsim_decide PUBLIC OpenMP::OpenMP_CXX)
