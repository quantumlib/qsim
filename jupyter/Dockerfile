# Base OS
FROM centos:8
USER root
# Install baseline

RUN yum -y update && \
	yum install -y epel-release && \
	yum group install -y "Development Tools" && \
	yum install -y python3-devel cmake python3-pip git && \
	python3 -m pip install --upgrade pip && \
	python3 -m pip install cirq && \
	python3 -m pip install cirq[contrib] && \
	python3 -m pip install qsimcirq && \
	python3 -m pip install jupyterlab && \
    python3 -m pip install jupyter_http_over_ws && \
    jupyter serverextension enable --py jupyter_http_over_ws && \
	cd / && \
	git clone https://github.com/quantumlib/qsim.git

RUN groupadd -g 412 appuser && \
    useradd -m -r -u 412 -g appuser appuser 
USER appuser

CMD ["jupyter-notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
