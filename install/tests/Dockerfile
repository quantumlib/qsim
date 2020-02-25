# Base OS
FROM debian

# Install requirements
RUN apt-get update
RUN apt-get install -y python3-dev python3-pybind11 python3-pytest python3-pip
RUN apt-get install -y cmake git

COPY ./requirements.txt /qsim/requirements.txt

# TODO: remove once live qsimcirq package is used.
RUN pip3 install -r /qsim/requirements.txt
# TODO: replace with the live qsimcirq pypi package once it is available.
RUN pip3 install -i https://test.pypi.org/simple/ qsimcirq

COPY ./interfaces/tests/ /qsim/interfaces/tests/

WORKDIR /qsim/

ENTRYPOINT python3 -m pytest ./interfaces/tests/