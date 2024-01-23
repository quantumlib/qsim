# Base OS
FROM debian

# Install requirements
RUN apt-get update
RUN apt-get install -y python3-dev python3-pip python3-venv
RUN apt-get install -y cmake git

# Create venv to avoid collision between system packages (e.g. numpy) and Cirq's deps.
RUN python3 -m venv test_env

# Activate venv.
ENV PATH="test_env/bin:$PATH"

COPY ./ /qsim/
RUN pip3 install /qsim/

# Run test in a non-qsim directory
COPY ./qsimcirq_tests/ /test-install/

WORKDIR /test-install/

ENTRYPOINT python3 -m pytest ./
