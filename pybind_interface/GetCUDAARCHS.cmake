# Copyright 2025 Google LLC. All Rights Reserved.
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


# Check whether the user has provided info about the GPU(s) installed
# on their system. If not, try to determine what it is automaticaly.
if(CMAKE_CUDA_ARCHITECTURES)
    # CMake 3.18+ sets this variable from $CUDAARCHS automatically.
    message(STATUS "qsim: using CUDA architectures "
                   "${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Compile for all supported major and minor real architectures, and the
    # highest major virtual architecture.
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
