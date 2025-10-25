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

# Base OS
FROM ubuntu:24.04 AS qsim-base

# Allow passing this variable in from the outside.
ARG CUDA_PATH
ENV PATH="$CUDA_PATH/bin:$PATH"

# Update package list & install some basic tools we'll need.
# hadolint ignore=DL3009,DL3008
RUN apt-get update && \
    apt-get install -y make g++ wget git --no-install-recommends && \
    apt-get install -y python3-dev python3-pip python3-venv --no-install-recommends

# Ubuntu 24's version of CMake is 3.28. We need a newer version.
RUN apt-get remove --purge --auto-remove cmake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh && \
    sh cmake-3.28.1-linux-x86_64.sh --prefix=/usr/local --skip-license

# Copy relevant files for simulation.
COPY ./Makefile /qsim/Makefile
COPY ./apps/ /qsim/apps/
COPY ./circuits/ /qsim/circuits/
COPY ./lib/ /qsim/lib/
COPY ./pybind_interface/ /qsim/lib/
COPY ./qsimcirq_tests/ /qsim/qsimcirq_tests/
COPY ./requirements.txt /qsim/requirements.txt
COPY ./dev-requirements.txt /qsim/dev-requirements.txt

# Create venv to avoid collision between system packages and what we install.
RUN python3 -m venv --upgrade-deps test_env

# Activate venv.
ENV PATH="/test_env/bin:$PATH"

# Install qsim requirements.
# hadolint ignore=DL3042
RUN python3 -m pip install -r /qsim/requirements.txt && \
    python3 -m pip install -r /qsim/dev-requirements.txt

# Compile qsim.
WORKDIR /qsim/
RUN make -j qsim

ENTRYPOINT ["/qsim/apps/qsim_base.x"]
