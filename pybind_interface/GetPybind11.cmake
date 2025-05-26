include(FetchContent)

set(MIN_PYBIND_VERSION "2.13.6")
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG "v${MIN_PYBIND_VERSION}"
)
FetchContent_GetProperties(pybind11)
find_package(pybind11 "${MIN_PYBIND_VERSION}" CONFIG)

if (pybind11_FOUND)
  message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
  # The pybind11_add_module doesn't correctly set the CXX_INCLUDES properly if a system pybind11 is found.
  # Using `include_directories(${pybind11_INCLUDE_DIRS})` doesn't result in anything in
  # CXX_INCLUDES. e.g., `pybind_interface/basic/CMakeFiles/qsim_basic.dir/flags.make` would only
  # have `CXX_INCLUDES = -isystem $PREFIX/include/python3.11` and would miss `$PREFIX/include`.
  # This problem would result in `fatal error: pybind11/complex.h: No such file or directory`
  # This is a hack to get around that by passing `-I/path/to/include` to CXX_FLAGS
  # Iterate over each include directory and add it as a compile option
  foreach(INCLUDE_DIR ${pybind11_INCLUDE_DIRS})
    add_compile_options("-I${INCLUDE_DIR}")
  endforeach()
endif()

if((NOT pybind11_FOUND) AND (NOT pybind11_POPULATED)) # check first on system path, then attempt git fetch
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
