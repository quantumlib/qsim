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
if(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
    message(STATUS "qsim: using CUDA archs string $ENV{CUDAARCHS}")
else()
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            COMMAND tr -d .
            OUTPUT_VARIABLE DETECTED_ARCHS
            RESULT_VARIABLE NVIDIA_SMI_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NVIDIA_SMI_RESULT EQUAL 0 AND DETECTED_ARCHS)
            # The command was successful. The output may contain
            # multiple lines if there are multiple GPUs.
            string(REPLACE "\n" ";" ARCHS_LIST "${DETECTED_ARCHS}")
            set(CMAKE_CUDA_ARCHITECTURES "${ARCHS_LIST}")
        else()
            message(FATAL_ERROR "nvidia-smi failed or returned no GPU"
                                " architectures. Please check your"
                                " NVIDIA driver installation and your"
                                " command search PATH variable.")
        endif()
    else()
        message(FATAL_ERROR "nvidia-smi not found. Cannot autodetect"
                            " the current GPU architecture(s). Please"
                            " set the environment variable CUDAARCHS"
                            " to an appropriate value and try again.")
    endif()
endif()

