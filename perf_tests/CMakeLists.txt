cmake_minimum_required(VERSION 3.25)
project("BooleanNet-PerfTests" CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

find_package(CUDAToolkit REQUIRED)

add_executable(perf_test 
    perf_test.cpp
    ../src/BooleanNetCUDA.cu
)

target_link_libraries(perf_test PRIVATE CUDA::cudart)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(perf_test PRIVATE -Wno-deprecated-declarations)
endif()

if (CMAKE_CUDA_COMPILER)
    target_compile_options(perf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
endif()