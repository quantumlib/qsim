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
