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
    sha256 = "ab78fa3f912d44d38b785ec011a25f26512aaedc5291f51f3807c592b506d33a",
    strip_prefix = "googletest-58d77fa8070e8cec2dc1ed015d66b454c8d78850",
    url = "https://github.com/google/googletest/archive/58d77fa8070e8cec2dc1ed015d66b454c8d78850.zip",
)

# Required for testing compatibility with TF Quantum:
# https://github.com/tensorflow/quantum
http_archive(
    name = "org_tensorflow",
    sha256 = "447cdb65c80c86d6c6cf1388684f157612392723eaea832e6392d219098b49de",
    strip_prefix = "tensorflow-2.13.0",
    url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.zip",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()


EIGEN_COMMIT = "3bb6a48d8c171cf20b5f8e48bfb4e424fbd4f79e"
EIGEN_SHA256 = "eca9847b3fe6249e0234a342b78f73feec07d29f534e914ba5f920f3e09383a3"


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
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
)

load("//third_party/cuquantum:cuquantum_configure.bzl", "cuquantum_configure")

cuquantum_configure(name = "local_config_cuquantum")
