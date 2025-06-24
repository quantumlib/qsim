# Base OS
FROM qsim

# Copy relevant files
COPY ./pybind_interface/ /qsim/pybind_interface/
COPY ./qsimcirq/ /qsim/qsimcirq/
COPY ./qsimcirq_tests/ /qsim/qsimcirq_tests/

WORKDIR /qsim/

# Build pybind code early to cache the results
RUN make -C /qsim/ pybind

# Compile and run qsim tests
ENTRYPOINT ["make", "-C", "/qsim/", "run-py-tests"]
