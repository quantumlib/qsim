# Base OS
FROM qsim

# Install additional requirements
RUN apt-get install -y python3-dev python3-pybind11 python3-pip python3-venv

# Create venv to avoid collision between system packages (e.g. numpy) and Cirq's deps.
RUN python3 -m venv test_env

# Activate venv.
ENV PATH="test_env/bin:$PATH"

# break-system-packages flag to override system packages (e.g. numpy) with Cirq's deps.
RUN pip3 install --prefer-binary cirq-core

# Copy relevant files
COPY ./pybind_interface/ /qsim/pybind_interface/
COPY ./qsimcirq/ /qsim/qsimcirq/
COPY ./qsimcirq_tests/ /qsim/qsimcirq_tests/

WORKDIR /qsim/

# Build pybind code early to cache the results
RUN make -C /qsim/ pybind

# Install pytest
RUN pip3 install pytest

# Compile and run qsim tests
ENTRYPOINT make -C /qsim/ run-py-tests
