cmake_minimum_required(VERSION 3.11)
project(qsim)

set(CMAKE_CXX_FLAGS "-O3 -mavx2 -mfma -fopenmp")

pybind11_add_module(qsim pybind_interface/pybind_main.cpp)
