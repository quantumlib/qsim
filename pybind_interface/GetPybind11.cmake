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

include(FetchContent)

set(MIN_PYBIND_VERSION "2.13.6")

# Suppress warning "Compatibility with CMake < 3.10 will be removed ..." coming
# from Pybind11. Not ideal, but avoids wasting time trying to find the cause.
# TODO(mhucka): remove the settings when pybind11 updates its CMake files
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Disable CMake deprecation warnings" FORCE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG "v${MIN_PYBIND_VERSION}"
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_WARN_DEPRECATED ON CACHE BOOL "Reenable CMake deprecation warnings" FORCE)
