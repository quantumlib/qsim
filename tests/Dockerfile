# Base OS
FROM qsim-base AS qsim-cxx-tests

# Copy relevant files
COPY ./tests/ /qsim/tests/

WORKDIR /qsim/

# Compile and run qsim tests
ENTRYPOINT ["make", "-C", "/qsim/", "run-cxx-tests"]
