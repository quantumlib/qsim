cmake_minimum_required(VERSION 3.11)
project(qsim)


IF (WIN32)
    set(CMAKE_CXX_FLAGS "/arch:AVX512 /O2 /openmp")
ELSE()
    set(CMAKE_CXX_FLAGS "-mavx512f -O3 -fopenmp")
ENDIF()

if(APPLE)
    set(CMAKE_CXX_STANDARD 14)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.2.4
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
pybind11_add_module(qsim_avx512 pybind_main_avx512.cpp)
