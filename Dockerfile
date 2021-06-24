# Base OS
FROM debian

# Install baseline
RUN apt-get update
RUN apt-get install -y g++ make wget

# Copy relevant files for simulation
COPY ./Makefile /qsim/Makefile
COPY ./apps/ /qsim/apps/
COPY ./circuits/ /qsim/circuits/
COPY ./lib/ /qsim/lib/

WORKDIR /qsim/

# Compile qsim
RUN make qsim

ENTRYPOINT ["/qsim/apps/qsim_base.x"]
