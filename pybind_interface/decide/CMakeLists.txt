cmake_minimum_required(VERSION 3.11)

execute_process(COMMAND which nvcc OUTPUT_VARIABLE has_nvcc)
if(has_nvcc STREQUAL "")
  project(qsim)
else()
  project(qsim LANGUAGES CXX CUDA)
endif()

IF (WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /openmp")
ELSE()
    set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
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

if(has_nvcc STREQUAL "")
  pybind11_add_module(qsim_decide decide.cpp)
else()
  find_package(PythonLibs 3.6 REQUIRED)
  find_package(CUDA REQUIRED)

  include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)

  cuda_add_library(qsim_decide MODULE decide.cpp)

  if(DEFINED ENV{CUQUANTUM_ROOT})
      target_compile_options(qsim_decide PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-D__CUSTATEVEC__>
      )
  endif()

  set_target_properties(qsim_decide PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
  set_source_files_properties(decide.cpp PROPERTIES LANGUAGE CUDA)
endif()
