# Docker

[Docker](https://docker.com) can be used to run qsim tests in a self-contained
environment, regardless of your local operating system.

## Build Docker Images

To build qsim and run all the tests:

```
# docker-compose up --build
```

`docker-compose` will create the `qsim-cxx-tests` image and automatically run
all tests. To build without running tests, simply run:

```
# docker-compose build
```

## Run Tests

Once the `qsim-cxx-tests` image is created, use the following command to run
all C++ tests:

```
# docker run -ti --rm qsim-cxx-tests
```

## Run Simulations

TODO: Support running simulations from Docker.
