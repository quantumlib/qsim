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

"""Repository rule to probe compiler support for SFrame flags."""

def _compiler_probe_impl(repository_ctx):
    # Try to determine the compiler. Prefer CC from environment.
    cc = repository_ctx.os.environ.get("CC", "c++")

    # Run a test compilation to test if the flag is recognized.
    # This is hacky and I wish there was a better way.
    res = repository_ctx.execute([
        cc,
        "-Wa,--gsframe=no",
        "-c",
        "-x",
        "c++",
        "/dev/null",
        "-o",
        "/dev/null",
    ])

    supports_gsframe = (res.return_code == 0)

    repository_ctx.file("BUILD.bazel", "package(default_visibility = ['//visibility:public'])\n")
    repository_ctx.file("compiler_config.bzl", "SUPPORTS_GSFRAME = %s\n" % supports_gsframe)

compiler_probe = repository_rule(
    implementation = _compiler_probe_impl,
    local = True,
    environ = ["CC", "PATH"],
)
