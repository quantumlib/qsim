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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuquantum_headers",
    linkstatic = 1,
    srcs = [":cuquantum_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libcuquantum",
    srcs = [
        ":libcustatevec.so",
    ],
    visibility = ["//visibility:public"],
)

%{CUQUANTUM_HEADER_GENRULE}
%{CUSTATEVEC_SHARED_LIBRARY_GENRULE}
