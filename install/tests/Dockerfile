# Base OS
FROM ubuntu:24.04

# Allow passing this variable in from the outside.
ARG CUDA_PATH
ENV PATH="$CUDA_PATH/bin:$PATH"

# Update package list & install some basic tools we'll need.
RUN apt-get update
RUN apt-get install -y python3-dev python3-pip python3-venv make g++ wget git

# Ubuntu 24's version of CMake is 3.28. We need a newer version.
RUN apt-get remove --purge --auto-remove cmake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.sh
RUN sh cmake-3.31.7-linux-x86_64.sh --prefix=/usr/local --skip-license

# Copy qsim files from the outside-Docker location to an inside-Docker location.
COPY ./ /qsim/

# Switch to that location.
WORKDIR /qsim/

# Create venv to avoid collision between system packages and what we install.
RUN python3 -m venv --upgrade-deps /qsim/test_env 

# Activate the venv.
ENV PATH="/qsim/test_env/bin:$PATH"

# Install qsim from sources.
# Note: use pip3 here, not python3 -m pip. We need to make sure to get the pip
# installed inside the venv, because that one has the correct sys.path.
RUN pip3 install -v /qsim/

# Copy the tests to a non-qsim directory.
COPY ./qsimcirq_tests/ /test-install/

# Run the tests from that location.
WORKDIR /test-install/
ENTRYPOINT ["python3", "-m", "pytest", "./"]
