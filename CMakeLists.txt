cmake_minimum_required(VERSION 3.10)

project(MiniCNN)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)


if(MSVC)
    add_compile_options(/O2 /openmp)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

add_executable(lenet lenet.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lenet OpenMP::OpenMP_CXX)
endif()

# Optionally use OpenBLAS for faster GEMM
option(USE_OPENBLAS "Link against OpenBLAS if available" OFF)
if (USE_OPENBLAS)
    find_library(OPENBLAS_LIB NAMES openblas libopenblas)
    if (OPENBLAS_LIB)
        message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
        target_link_libraries(lenet PRIVATE ${OPENBLAS_LIB})
        target_compile_definitions(lenet PRIVATE HAVE_CBLAS=1)
    else()
        message(WARNING "OpenBLAS requested but not found. Building without it.")
    endif()
else()
    message(STATUS "Building without OpenBLAS (USE_OPENBLAS=OFF). Using internal GEMM fallback.")
endif()