# Base OS
FROM debian

# Install baseline
RUN apt-get update
RUN apt-get install -y g++ make cmake

# Copy relevant files
COPY ./ /qsim/

WORKDIR /qsim/

# Compile qsim
ENTRYPOINT make -C /qsim/tests/ run-all