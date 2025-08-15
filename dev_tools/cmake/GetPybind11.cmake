# Copyright 2025 Google LLC
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

if(NOT pybind11_FOUND)
    set(MIN_PYBIND_VERSION "2.13.6")

    find_package(
        pybind11
        CONFIG
        HINTS "${Python3_SITELIB}"
        NO_POLICY_SCOPE)

    # qsim's requirements.txt and setup.py both include a requirement for
    # "pybind11[global]", so the Pybind11 CMake plugin should be found no matter
    # whether the user is doing a "pip install qsim" or a local build. Still, we
    # want to be sure, and also want to make sure to get the min version we need.
    if(NOT pybind11_FOUND OR ${pybind11_VERSION} VERSION_LESS ${MIN_PYBIND_VERSION})
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG "v${MIN_PYBIND_VERSION}"
            OVERRIDE_FIND_PACKAGE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
endif()
