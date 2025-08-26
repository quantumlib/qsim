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
FROM ubuntu:24.04
USER root

# Install baseline
RUN apt-get -y update && \
        apt-get install -y python3-dev python3-pip git && \
        python3 -m pip install --upgrade pip && \
        python3 -m pip install cirq && \
        python3 -m pip install cirq[contrib] && \
        python3 -m pip install qsimcirq && \
        python3 -m pip install jupyterlab && \
    python3 -m pip install jupyter_http_over_ws && \
    jupyter serverextension enable --py jupyter_http_over_ws && \
        cd / && \
        git clone https://github.com/quantumlib/qsim.git

RUN  jupyter serverextension enable --py jupyter_http_over_ws

CMD ["jupyter-notebook", "--port=8888", "--no-browser",\
      "--ip=0.0.0.0", "--allow-root", \
          "--NotebookApp.allow_origin='*'", \
      "--NotebookApp.port_retries=0", \
          "--NotebookApp.token=''"]
