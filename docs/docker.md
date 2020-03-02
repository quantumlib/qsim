# Docker

[Docker](https://docker.com) can be used to run qsim tests in a self-contained
environment, regardless of your local operating system.

**NOTE:** Docker requires root access to run. If you see errors when running
the commands below, you may need to call them with `sudo`.

## Build Docker Images

Prior to building with Docker, make sure that your repository is clean and all
submodules are up-to-date. The following commands should accomplish this:

```
make clean
git submodule update --init --recursive
```

To build qsim and run all the tests:

```
# docker-compose up --build
```

`docker-compose` will create the `qsim`, `qsim-cxx-tests`, and `qsim-py-tests`
images and automatically run all tests. A successful run should have the
following messages somewhere in the logs:

```
qsim-cxx-tests exited with code 0
qsim-py-tests exited with code 0
```

To build without running tests, simply run:

```
# docker-compose build
```

## Run Simulations

Once the `qsim` image is created, it can be used to run the `qsim_base.x`
simulation binary with the following command:

```
# docker run -ti --rm -v $PWD/circuits:/qsim/circuits:ro \
                      qsim:latest -c /qsim/circuits/circuit_q24
```

The flag `-v [orig]:[dest]:[attr]` is required to allow access to the host
folders from within the `qsim` image. This can be omitted only if the circuit
to be simulated has not been modified since the image was created.

## Run C++ Tests

Once the `qsim-cxx-tests` image is created, use the following command to run
all C++ tests:

```
# docker run -ti --rm qsim-cxx-tests
```

## Run Python Tests

Once the `qsim-py-tests` image is created, use the following command to run
all Python tests:

```
# docker run -ti --rm qsim-py-tests
```
