# Copyright 2018 Google LLC
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

import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# qsimcirq/_version.py contains the source of truth for the version nhumber.
__version__ = runpy.run_path("qsimcirq/_version.py")["__version__"]
assert __version__, "The version string must not be empty"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            from packaging.version import parse

            cmake_version = parse(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < parse("3.28.0"):
                raise RuntimeError("CMake >= 3.28.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_include_dir = sysconfig.get_path("include")
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPYTHON_INCLUDE_DIR=" + python_include_dir,
        ]

        if shutil.which("nvcc") is not None:
            cmake_args += [
                "-DCMAKE_CUDA_COMPILER=nvcc",
            ]

        # Append additional CMake arguments from the environment variable.
        # This is e.g. used by cibuildwheel to force a certain C++ standard.
        additional_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if additional_cmake_args:
            cmake_args += additional_cmake_args.split()

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j"]

        if platform.system() == "Darwin":
            homebrew_x86 = "/usr/local/opt/llvm@19/bin"
            homebrew_arm = "/opt/homebrew/opt/llvm@19/bin"
            # Add clang
            if shutil.which("clang") is not None:  # Always prefer from PATH
                cmake_args.append("-DCMAKE_C_COMPILER=clang")
            elif os.path.exists(f"{homebrew_x86}/clang"):
                cmake_args.append(f"-DCMAKE_C_COMPILER={homebrew_x86}/clang")
            elif os.path.exists(f"{homebrew_arm}/clang"):
                cmake_args.append(f"-DCMAKE_C_COMPILER={homebrew_arm}/clang")

            # Add clang++
            if shutil.which("clang++") is not None:  # Always prefer from PATH
                cmake_args.append("-DCMAKE_CXX_COMPILER=clang++")
            elif os.path.exists(f"{homebrew_x86}/clang++"):
                cmake_args.append(f"-DCMAKE_CXX_COMPILER={homebrew_x86}/clang++")
            elif os.path.exists(f"{homebrew_arm}/clang++"):
                cmake_args.append(f"-DCMAKE_CXX_COMPILER={homebrew_arm}/clang++")

        if shutil.which("hipcc") is not None:
            cmake_args += [
                "-DCMAKE_C_COMPILER=hipcc",
                "-DCMAKE_CXX_COMPILER=hipcc",
            ]

        env = os.environ.copy()
        cxxflags = env.get("CXXFLAGS", "")
        env["CXXFLAGS"] = (
            f'{cxxflags} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose"] + build_args,
            cwd=self.build_temp,
        )


setup(
    ext_modules=[
        CMakeExtension("qsimcirq/qsim_avx512"),
        CMakeExtension("qsimcirq/qsim_avx2"),
        CMakeExtension("qsimcirq/qsim_sse"),
        CMakeExtension("qsimcirq/qsim_basic"),
        CMakeExtension("qsimcirq/qsim_cuda"),
        CMakeExtension("qsimcirq/qsim_custatevec"),
        CMakeExtension("qsimcirq/qsim_custatevecex"),
        CMakeExtension("qsimcirq/qsim_decide"),
        CMakeExtension("qsimcirq/qsim_hip"),
    ],
    cmdclass={"build_ext": CMakeBuild},
)
