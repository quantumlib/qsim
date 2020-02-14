# Base OS
FROM qsim

# Install additional requirements
RUN apt-get install -y cmake git

# Copy relevant files
COPY ./tests/ /qsim/tests/

WORKDIR /qsim/

# Compile and run qsim tests
ENTRYPOINT make -C /qsim/ run-cxx-tests