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

# TODO: remove reliance on getopt (unistd.h) to allow apps to run on Windows.

cc_binary(
    name = "qsim_base",
    srcs = ["qsim_base.cc"],
    deps = [
        "//lib:run_qsim_lib",
    ],
    data = ["//circuits:circuit_q24"],
)

cc_binary(
    name = "qsim_von_neumann",
    srcs = ["qsim_von_neumann.cc"],
    deps = [
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsim_amplitudes",
    srcs = ["qsim_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsim_lib",
    ],
)

cc_binary(
    name = "qsimh_base",
    srcs = ["qsimh_base.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)

cc_binary(
    name = "qsimh_amplitudes",
    srcs = ["qsimh_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_qsimh_lib",
    ],
)
