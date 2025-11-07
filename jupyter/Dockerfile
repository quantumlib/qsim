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

FROM ubuntu:24.04 AS qsimcirq

# hadolint ignore=DL3002
USER root

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Newer Ubuntus adopted PEP 668, which blocks pip from installing into system
# locations, but Docker is already an isolated environment, so we don't mind.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN python3 -m pip install --no-cache-dir \
    cirq~=1.0 \
    cirq[contrib]~=1.0 \
    qsimcirq~=0.1 \
    notebook==6.4.8 \
    jupyterlab==4.4.10 \
    jupyter-contrib-nbextensions==0.7.0 \
    jupyter-nbextensions-configurator==0.6.4 \
    jupyter-http-over-ws==0.0.8

WORKDIR /

RUN git clone https://github.com/quantumlib/qsim.git

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable varInspector/main
RUN jupyter serverextension enable --py jupyter_http_over_ws

CMD ["jupyter-notebook", "--port=8888", "--no-browser",\
      "--ip=0.0.0.0", "--allow-root", \
          "--NotebookApp.allow_origin='*'", \
      "--NotebookApp.port_retries=0", \
          "--NotebookApp.token=''"]
