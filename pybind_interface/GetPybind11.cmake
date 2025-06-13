include(FetchContent)

set(MIN_PYBIND_VERSION "2.13.6")
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG "v${MIN_PYBIND_VERSION}"
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(pybind11)
