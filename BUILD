# Copyright 2026 Google LLC
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

package(default_visibility = ["//visibility:public"])

# Define configurations for build with AVX and/or SSE support.

config_setting(
    name = "avx_requested",
    values = {"define": "qsim_avx=true"},
)

config_setting(
    name = "sse_requested",
    values = {"define": "qsim_sse=true"},
)

config_setting(
    name = "native_requested",
    values = {"define": "qsim_native=true"},
)
