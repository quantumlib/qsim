cmake_minimum_required(VERSION 3.11)
project(qsim)

ADD_SUBDIRECTORY(pybind_interface/sse)
ADD_SUBDIRECTORY(pybind_interface/avx512)
ADD_SUBDIRECTORY(pybind_interface/avx2)
ADD_SUBDIRECTORY(pybind_interface/basic)
ADD_SUBDIRECTORY(pybind_interface/decide)
