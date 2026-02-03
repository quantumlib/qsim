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

"""Test setup.py code."""

import os
import subprocess
import sys


def run_setup(cmake_args, root_dir):
    env = os.environ.copy()
    env["CMAKE_ARGS"] = cmake_args

    # We use 'build_ext' but it might fail if CMake is not installed or other
    # reasons. We just want to see if it raises the RuntimeError we added in
    # build_extension.
    try:
        # Using sys.executable to run setup.py with the same Python executable.
        return subprocess.run(
            [sys.executable, "setup.py", "build_ext"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(root_dir),
        )
    except FileNotFoundError as e:
        # This Dummy class emulates what subprocess.run() returns.
        class Dummy:
            stderr = f"Command or file not found: {e}"
            returncode = 1

        return Dummy()
    except PermissionError as e:

        class Dummy:
            stderr = f"Permission denied: {e}"
            returncode = 1

        return Dummy()
    except OSError as e:

        class Dummy:
            stderr = f"OS error occurred: {e}"
            returncode = 1

        return Dummy()
    except Exception as e:
        # Fallback for unexpected errors in subprocess.run itself
        class Dummy:
            stderr = f"Unexpected error: {str(e)}"
            returncode = 1

        return Dummy()


def test_valid_cmake_args(pytestconfig):
    res = run_setup("-DCMAKE_CXX_STANDARD=17", pytestconfig.rootpath)
    # If it fails, it shouldn't be because of our validation.
    assert "is invalid; all arguments must begin with a dash (-)." not in res.stderr


def test_invalid_cmake_args_no_dash(pytestconfig):
    res = run_setup("NOT_A_FLAG", pytestconfig.rootpath)
    assert "is invalid; all arguments must begin with a dash (-)." in res.stderr


def test_invalid_cmake_args_malicious(pytestconfig):
    res = run_setup("-DVAR=VAL ; rm -rf /", pytestconfig.rootpath)
    assert "is invalid; all arguments must begin with a dash (-)." in res.stderr
