cmake_minimum_required(VERSION 3.25)
project("BooleanNet-PerfTests" CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")

# Configure Alpaka Options BEFORE adding subdirectory
set(alpaka_DEP_CUDA ON)
set(CMAKE_CUDA_ARCHITECTURES "native") 

# Enable CPU backends
set(alpaka_EXEC_CpuSerial ON)      
set(alpaka_DEP_OMP ON)
set(alpaka_EXEC_CpuOmpBlocks ON)

# Enable TBB if available (optional)
# set(alpaka_DEP_TBB ON)
# set(alpaka_EXEC_TbbBlocks ON)

# Add Alpaka
add_subdirectory("../../common/alpaka3" "${CMAKE_BINARY_DIR}/alpaka_build")
find_package(alpaka REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Executable
add_executable(perf_test perf_test.cpp)

target_link_libraries(perf_test PUBLIC alpaka::alpaka)
alpaka_finalize(perf_test)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(perf_test PRIVATE -Wno-deprecated-declarations)
endif()

if (CMAKE_CUDA_COMPILER)
    target_compile_options(perf_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
endif()