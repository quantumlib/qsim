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

if(WIN32)
    # Visual Studio 17.11.5 added /arch:SSE4.2 in Oct. 2024. Try to use
    # that if it's available.
    check_cxx_compiler_flag("/arch:SSE4.2" WIN32_SSE4_AVAILABLE)
    if(WIN32_SSE4_AVAILABLE)
        add_compile_options(/arch:SSE4.2)
    else()
        # VS 2022 docs say that using /arch:SSE2 will make the auto-vectorizer
        # emit 4.2 instructions when available. So, resort to this if we can.
        check_cxx_compiler_flag("/arch:SSE2" WIN32_SSE2_AVAILABLE)
        if(WIN32_SSE2_AVAILABLE)
            add_compile_options(/arch:SSE2)
        else()
            # Some sources say that to get SSE4.1, you typically use /arch:AVX.
            add_compile_options(/arch:AVX)
        endif()
    endif()
else()
    add_compile_options(-msse4.1)
endif()

INCLUDE(../GetPybind11.cmake)
pybind11_add_module(qsim_sse pybind_main_sse.cpp)

target_link_libraries(qsim_sse PUBLIC OpenMP::OpenMP_CXX)
