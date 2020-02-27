# Base OS
FROM qsim

# Install additional requirements
RUN apt-get install -y python3-dev python3-pybind11 python3-pytest python3-pip
RUN pip3 install cirq

# Copy relevant files
COPY ./pybind_interface/ /qsim/pybind_interface/
COPY ./qsimcirq/ /qsim/qsimcirq/
COPY ./interfaces/ /qsim/interfaces/

WORKDIR /qsim/

# Build pybind code early to cache the results
RUN make -C /qsim/ pybind

# Compile and run qsim tests
ENTRYPOINT make -C /qsim/ run-py-tests