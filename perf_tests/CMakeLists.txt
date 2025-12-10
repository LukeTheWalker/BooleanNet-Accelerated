cmake_minimum_required(VERSION 3.25)

project("BooleanNet-PerfTests" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")

# --- Alpaka Configuration ---
# set(alpaka_DEP_HIP ON)
# set(CMAKE_HIP_ARCHITECTURES "gfx942")
# set(alpaka_DEP_CUDA ON)


# Add Alpaka (builds from source)
add_subdirectory("../../common/alpaka3" "${CMAKE_BINARY_DIR}/alpaka_build")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# --- Define Executable ---
add_executable(perf_test perf_test.cpp)
target_link_libraries(perf_test PUBLIC alpaka::alpaka) # Keep alpaka here
alpaka_finalize(perf_test)
