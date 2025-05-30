# Base OS
FROM ubuntu:24.04

# Update package list & install some basic tools we'll need.
RUN apt update
RUN apt install -y make g++ wget git

# The default version of CMake is 3.28. Get a newer version from Kitware.
RUN apt remove --purge --auto-remove cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.sh
RUN sh cmake-3.31.7-linux-x86_64.sh --prefix=/usr/local --skip-license

# Copy relevant files for simulation
COPY ./Makefile /qsim/Makefile
COPY ./apps/ /qsim/apps/
COPY ./circuits/ /qsim/circuits/
COPY ./lib/ /qsim/lib/

# Copy Python requirements file for other images based on this one.
COPY ./requirements.txt /qsim/requirements.txt

# Compile qsim.
WORKDIR /qsim/
RUN make qsim

ENTRYPOINT ["/qsim/apps/qsim_base.x"]
