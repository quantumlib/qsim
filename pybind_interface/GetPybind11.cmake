set(MIN_PYBIND_VERSION "2.13.6")

# First, try to find pybind11 using the modern find_package() way.
find_package(pybind11 ${MIN_PYBIND_VERSION} QUIET)

# If that doesn't work, try to use pybind11-config to find the CMake directory.
if(NOT pybind11_FOUND)
  find_program(PYBIND11_CONFIG_EXECUTABLE pybind11-config)

  if(PYBIND11_CONFIG_EXECUTABLE)
    # Get the CMake directory from pybind11-config
    execute_process(
      COMMAND ${PYBIND11_CONFIG_EXECUTABLE} --cmakedir
      OUTPUT_VARIABLE pybind11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Add the discovered path to the CMake search paths and try again.
    if(pybind11_CMAKE_DIR AND IS_DIRECTORY ${pybind11_CMAKE_DIR})
      list(APPEND CMAKE_PREFIX_PATH ${pybind11_CMAKE_DIR})
      find_package(pybind11 ${MIN_PYBIND_VERSION} QUIET)
    endif()
  endif()
endif()

# If pybind11 is still not found, use FetchContent to download it.
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found. Fetching from source.")
  include(FetchContent)

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG "v${MIN_PYBIND_VERSION}"
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(pybind11)
else()
  message(STATUS "Found pybind11: ${pybind11_VERSION} (found version: ${pybind11_VERSION})")
endif()
