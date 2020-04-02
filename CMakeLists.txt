cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_FLAGS "-O3 -mavx2 -mfma -fopenmp -shared `python3 -m pybind11 --includes` `python3-config --ldflags`")

add_executable(qsim pybind_interface/pybind_main.cpp)
