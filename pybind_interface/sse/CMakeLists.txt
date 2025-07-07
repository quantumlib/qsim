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

# If the present module is included in the qsim build, it means that the
# top-level CMakeLists.txt found evidence that the hardware supports SSE4.1.
# However, the correct compiler flag still needs to be determined.
if(WIN32)
    # Some comments in forums suggest that for some programs, using arch=sse4.1
    # may provide better performance than using arch=sse4.2. It's not available
    # in newer VS but still try this flag first in case this compiler is older.
    check_cxx_compiler_flag("/arch:SSE4.1" WIN32_SSE4_AVAILABLE)
    if(WIN32_SSE4_AVAILABLE)
        add_compile_options(/arch:SSE4.1)
    else()
        # Visual Studio 17.11.5 added /arch:SSE4.2 in Oct. 2024. Try to use that
        # if it's available.
        check_cxx_compiler_flag("/arch:SSE4.2" WIN32_SSE4_AVAILABLE)
        if(WIN32_SSE4_AVAILABLE)
            add_compile_options(/arch:SSE4.2)
        else()
            # VS 2022 docs say /arch:SSE2 will make the auto-vectorizer emit 4.2
            # instructions when available. Use this if we get this far.
            check_cxx_compiler_flag("/arch:SSE2" WIN32_SSE2_AVAILABLE)
            if(WIN32_SSE2_AVAILABLE)
                add_compile_options(/arch:SSE2)
            else()
                # Although it's not guaranteed that AVX2 support guarantees
                # SSE4.1 too, it seems to be true in practice.
                if(CPU_SUPPORTS_AVX2)
                    # If we get here, it means elsewhere we'll have added avx2.
                    # Nothing more to do.
                else()
                    # Other options failed. Final fallback: in practice, if a
                    # CPU supports AVX, it is likely to also support SSE4.1.
                    add_compile_options(/arch:AVX)
                endif()
            endif()
        endif()
    endif()
else()
    add_compile_options(-msse4.1)
endif()

pybind11_add_module(qsim_sse pybind_main_sse.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qsim_sse PUBLIC pybind11::headers OpenMP::OpenMP_CXX)
endif()
