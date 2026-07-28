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


# ---------------------------------------------------------------------------
# rules_python — hermetic Python toolchain + pip dependencies
# ---------------------------------------------------------------------------

http_archive(
    name = "rules_python",
    sha256 = "4a02240a4a6a8b04077a7e49921949d2e6879d6f047be1e870d222de6709c7d1",
    strip_prefix = "rules_python-1.4.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/v1.4.1/rules_python-v1.4.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

# Register a hermetic Python interpreter (3.11 in this example; pick what qsim supports)
load("@rules_python//python:versions.bzl", "MINOR_MAPPING")
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python_3_11",
    python_version = "3.11",
)

load("@python_3_11//:defs.bzl", "interpreter")

# Parse locked requirements.
# pip_parse needs a lockfile, not a bare requirements.txt.
# We'll generate the lockfile in step 2.
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip",
    python_interpreter_target = interpreter,
    requirements_lock = "//:requirements_lock.txt",
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

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

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
