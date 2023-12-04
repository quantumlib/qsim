cmake_minimum_required(VERSION 3.11)

if(WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /openmp")
else()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

if(APPLE AND NOT APPLE_ARM)
    set(CMAKE_CXX_STANDARD 14)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

include(../GetPybind11.cmake)

# Configure based on the detected platform
if(has_nvcc)
    find_package(CUDA REQUIRED)
    cuda_add_library(qsim_decide MODULE decide.cpp)
    if(DEFINED ENV{CUQUANTUM_ROOT})
        target_compile_options(qsim_decide PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-D__CUSTATEVEC__>
        )
    endif()
    find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)
    set_target_properties(qsim_decide PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(qsim_decide OpenMP::OpenMP_CXX)
elseif(has_hipcc)
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")
    find_package(HIP REQUIRED)
    hip_add_library(qsim_decide MODULE decide.cpp)
    set_source_files_properties(decide.cpp PROPERTIES LANGUAGE HIP)
    find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter Development)
    include_directories(${PYTHON_INCLUDE_DIRS} ${pybind11_SOURCE_DIR}/include)
    set_target_properties(qsim_decide PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    target_link_libraries(qsim_decide PUBLIC OpenMP::OpenMP_CXX)
else()
    pybind11_add_module(qsim_decide decide.cpp)
    target_link_libraries(qsim_decide PUBLIC OpenMP::OpenMP_CXX)
endif()

