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

include_directories($ENV{CUQUANTUM_ROOT}/include)
link_directories($ENV{CUQUANTUM_ROOT}/lib $ENV{CUQUANTUM_ROOT}/lib64)

add_library(qsim_custatevec MODULE pybind_main_custatevec.cpp)
target_link_libraries(qsim_custatevec -lcustatevec -lcublas)

set_target_properties(qsim_custatevec PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_custatevec.cpp PROPERTIES LANGUAGE CUDA)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qsim_custatevec PUBLIC pybind11::headers OpenMP::OpenMP_CXX)
endif()

