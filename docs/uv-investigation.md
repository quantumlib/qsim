@'
# uv Dependency Management Investigation

## Purpose

This document investigates whether `uv` can be used for dependency management
and development workflows in qsim while retaining the existing native C++
build and Python packaging infrastructure.

The investigation covers development environment setup, dependency resolution,
testing, native extensions, package building, wheel installation, and CI
implications.

## Current qsim setup

qsim uses a `pyproject.toml` based Python package configuration with:

- setuptools as the build backend
- CMake and pybind11 for native extension builds
- setuptools-scm for version management
- Python dependency groups through `[dependency-groups]`
- `requirements.txt` as the source for runtime dependencies
- cibuildwheel for wheel builds
- Bazel for the C++ build and test workflow

The existing GitHub Actions workflows use pip for Python dependency
installation and pip caching through `actions/setup-python`.

## uv development environment

A Python 3.13 virtual environment was created with:

```text
uv venv .venv --python 3.13.9