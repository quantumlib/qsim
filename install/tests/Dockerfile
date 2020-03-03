# Base OS
FROM debian

# Install requirements
RUN apt-get update
RUN apt-get install -y python3-dev python3-pip
RUN apt-get install -y cmake git

COPY ./ /qsim/
RUN pip3 install /qsim/

# Run test in a non-qsim directory
COPY ./qsimcirq_tests/ /test-install/

WORKDIR /test-install/

ENTRYPOINT python3 -m pytest ./
