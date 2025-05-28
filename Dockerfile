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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base Docker image used to build images for running tests.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Base OS
FROM ubuntu:24.04

# Update package list & install some basic tools we'll need.
RUN apt update
RUN apt install -y make g++ wget

# The default version of CMake is 3.28. Get a newer version from Kitware.
RUN apt remove --purge --auto-remove cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.sh
RUN sh cmake-3.31.7-linux-x86_64.sh --prefix=/usr/local --skip-license

# Copy relevant files for simulation
COPY ./Makefile /qsim/Makefile
COPY ./apps/ /qsim/apps/
COPY ./circuits/ /qsim/circuits/
COPY ./lib/ /qsim/lib/

# Copy Python requirements file for other images that we build.
COPY ./requirements.txt /qsim/requirements.txt

# Compile qsim.
WORKDIR /qsim/
RUN make qsim

ENTRYPOINT ["/qsim/apps/qsim_base.x"]
