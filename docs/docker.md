# Docker

[Docker](https://docker.com) can be used to run qsim tests in a self-contained
environment, regardless of your local operating system.

Note that Docker is _not_ usually preinstalled on popular operating systems
such as MacOS, Windows, or Linux, so you will most likely have to install
Docker components yourself. Also, depending on the system, installing Docker
may require root access on your computer (e.g., using `sudo` on Linux).

The following packages are needed for the steps described on the rest of this
page:

*   `docker`
*   `docker-buildx`
*   `docker-compose`

## Build Docker Images

Prior to building with Docker, make sure that your local copy of the qsim
source code repository is clean and all submodules are up-to-date. The
following commands should accomplish this:

```
make clean
git submodule update --init --recursive
```

To build qsim and run all the tests:

```
docker compose up --build
```

`docker compose` will create the `qsim`, `qsim-cxx-tests`, and `qsim-py-tests`
images and automatically run all the tests for those targets. A successful run
should have the following messages somewhere in the logs:

```
qsim-cxx-tests exited with code 0
qsim-py-tests exited with code 0
```

To build the Docker image without running tests, simply run:

```
docker compose build
```

Note that currently (qsim version 0.23.0), the Docker build does not make use
of CUDA or GPUs.

## Run Simulations

Once the `qsim` Docker image is created, it can be used to run the
`qsim_base.x` simulation binary with the following command:

```
docker run -ti --rm -v $PWD/circuits:/qsim/circuits:ro \
    qsim:latest -c /qsim/circuits/circuit_q24
```

The flag `-v [orig]:[dest]:[attr]` is required to allow access to the host
folders from within the `qsim` image. This can be omitted only if the circuit
to be simulated has not been modified since the image was created.

## Run C++ Tests

Once the `qsim-cxx-tests` image is created, use the following command to run
all C++ tests:

```
docker run -ti --rm qsim-cxx-tests
```

## Run Python Tests

Once the `qsim-py-tests` image is created, use the following command to run
all Python tests:

```
docker run -ti --rm qsim-py-tests
```
