# Base OS
FROM qsim

# Install additional requirements
RUN apt-get install -y python3-dev python3-pybind11 python3-pytest python3-pip

# break-system-packages flag to override system packages (e.g. numpy) with Cirq's deps.
RUN pip3 install --break-system-packages --force-reinstall --prefer-binary cirq-core 

# Copy relevant files
COPY ./pybind_interface/ /qsim/pybind_interface/
COPY ./qsimcirq/ /qsim/qsimcirq/
COPY ./qsimcirq_tests/ /qsim/qsimcirq_tests/

WORKDIR /qsim/

# Build pybind code early to cache the results
RUN make -C /qsim/ pybind

# Compile and run qsim tests
ENTRYPOINT make -C /qsim/ run-py-tests
