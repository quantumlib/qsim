cmake_minimum_required(VERSION 3.18)
project(qsim LANGUAGES CXX HIP)

IF (WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /openmp")
ELSE()
    set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
ENDIF()

INCLUDE(../GetPybind11.cmake)
find_package(PythonLibs 3.7 REQUIRED)
find_package(HIP REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)

hip_add_library(qsim_hip MODULE pybind_main_hip.cpp)

set_target_properties(qsim_hip PROPERTIES
       PREFIX "${PYTHON_MODULE_PREFIX}"
       SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_hip.cpp PROPERTIES LANGUAGE HIP)
