include(FetchContent)

set(MIN_PYBIND_VERSION "2.2.4")
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG "v${MIN_PYBIND_VERSION}"
)
FetchContent_GetProperties(pybind11)
find_package(pybind11 "${MIN_PYBIND_VERSION}" CONFIG)
if((NOT pybind11_FOUND) AND (NOT pybind11_POPULATED)) # check first on system path, then attempt git fetch
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
