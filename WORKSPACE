load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    url = "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
)

# Required for testing compatibility with TF Quantum:
# https://github.com/tensorflow/quantum
http_archive(
    name = "org_tensorflow",
    sha256 = "e82f3b94d863e223881678406faa5071b895e1ff928ba18578d2adbbc6b42a4c",
    strip_prefix = "tensorflow-2.1.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.1.0.zip",
    ],
)


EIGEN_COMMIT = "12e8d57108c50d8a63605c6eb0144c838c128337"
EIGEN_SHA256 = "f689246e342c3955af48d26ce74ac34d21b579a00675c341721a735937919b02"


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
