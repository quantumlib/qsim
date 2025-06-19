include(FetchContent)

set(MIN_PYBIND_VERSION "2.13.6")

# Suppress warning "Compatibility with CMake < 3.10 will be removed ..." coming
# from Pybind11. Not ideal, but avoids wasting time trying to find the cause.
# TODO(mhucka): remove the settings when pybind11 updates its CMake files
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Disable CMake deprecation warnings" FORCE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG "v${MIN_PYBIND_VERSION}"
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_WARN_DEPRECATED ON CACHE BOOL "Reenable CMake deprecation warnings" FORCE)
