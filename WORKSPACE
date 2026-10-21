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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "platforms",
    sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
    ],
)

http_archive(
    name = "com_google_googletest",
    patch_args = ["-p1"],
    patches = ["//third_party:googletest.patch"],
    sha256 = "40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff",
    strip_prefix = "googletest-1.17.0",
    url = "https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip",
)

# Hermetic Python toolchain and pinned PyPI dependencies.

http_archive(
    name = "rules_python",
    sha256 = "ca77768989a7f311186a29747e3e95c936a41dffac779aff6b443db22290d913",
    strip_prefix = "rules_python-0.36.0",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/0.36.0/rules_python-0.36.0.tar.gz",
)

load(
    "@rules_python//python:repositories.bzl",
    "py_repositories",
    "python_register_toolchains",
)

py_repositories()

python_register_toolchains(
    name = "python_3_11",
    python_version = "3.11.9",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip",
    python_interpreter_target = "@python_3_11_host//:python",
    requirements_lock = "//third_party:requirements_lock.txt",
)

load("@pip//:requirements.bzl", "install_deps")

install_deps()

# Required for testing compatibility with TF Quantum:
# https://github.com/tensorflow/quantum
http_archive(
    name = "org_tensorflow",
    sha256 = "447cdb65c80c86d6c6cf1388684f157612392723eaea832e6392d219098b49de",
    strip_prefix = "tensorflow-2.13.0",
    url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.zip",
)

# qsim uses TensorFlow's CUDA repository rule, but does not directly depend on
# TensorFlow build targets. Avoid TensorFlow's broader workspace initialization,
# which runs a host-Python NumPy probe.
load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")

load("//dev_tools:compiler_probe.bzl", "compiler_probe")

compiler_probe(name = "local_compiler_config")

# https://gitlab.com/libeigen/eigen/-/releases/3.4.1
EIGEN_COMMIT = "b66188b5dfd147265bfa9ec47595ca0db72d21f5"

EIGEN_SHA256 = "2c167ff09e88a5261111bc2aa7f18ae2e78d73fd42339387532937b0c2629829"

http_archive(
    name = "eigen",
    build_file_content = """
cc_library(
  name = "eigen3",
  textual_hdrs = glob(["Eigen/**", "unsupported/**"]),
  visibility = ["//visibility:public"],
)
    """,
    sha256 = EIGEN_SHA256,
    strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
    urls = [
        "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
    ],
)

load("//third_party/cuquantum:cuquantum_configure.bzl", "cuquantum_configure")

cuquantum_configure(name = "local_config_cuquantum")
