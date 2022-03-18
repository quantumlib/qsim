# Base OS
FROM centos:8
USER root

# Centos 8 has reach end of life: https://www.centos.org/centos-linux-eol/
# Configuration must be loaded from the vault.
RUN pushd /etc/yum.repos.d/ && \
	sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
	sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
	popd

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

RUN  jupyter serverextension enable --py jupyter_http_over_ws

CMD ["jupyter-notebook", "--port=8888", "--no-browser",\
      "--ip=0.0.0.0", "--allow-root", \
	  "--NotebookApp.allow_origin='*'", \
      "--NotebookApp.port_retries=0", \
	  "--NotebookApp.token=''"]
