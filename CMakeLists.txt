cmake_minimum_required(VERSION 3.11)

# Check if we are on an Apple system and if the architecture is ARM
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # Set a variable to indicate we are on Apple ARM
    set(APPLE_ARM TRUE)
else()
    set(APPLE_ARM FALSE)
endif()

ADD_SUBDIRECTORY(pybind_interface/basic)
ADD_SUBDIRECTORY(pybind_interface/decide)

# You can then use this variable to control conditional compilation
if(APPLE_ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
else()
    execute_process(COMMAND which nvcc OUTPUT_VARIABLE has_nvcc)
    if(has_nvcc STREQUAL "")
        execute_process(COMMAND which hipcc OUTPUT_VARIABLE has_hipcc)
        if(has_hipcc STREQUAL "")
            project(qsim)
        else()
            project(qsim LANGUAGES CXX HIP)
            ADD_SUBDIRECTORY(pybind_interface/hip)
        endif()
    else()
        project(qsim LANGUAGES CXX CUDA)
        ADD_SUBDIRECTORY(pybind_interface/cuda)
        if(DEFINED ENV{CUQUANTUM_ROOT})
            ADD_SUBDIRECTORY(pybind_interface/custatevec)
        endif()
    endif()

    ADD_SUBDIRECTORY(pybind_interface/sse)
    ADD_SUBDIRECTORY(pybind_interface/avx512)
    ADD_SUBDIRECTORY(pybind_interface/avx2)
endif()
