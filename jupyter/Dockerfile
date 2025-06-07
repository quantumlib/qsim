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
