cmake_minimum_required(VERSION 3.25)
project("BooleanNet-PerfTests" CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

list(APPEND CMAKE_MODULE_PATH "/opt/rocm/hip/cmake")
find_package(hip REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(perf_test 
    perf_test.cpp
    ../src/BooleanNetHIP.hip
)

set_source_files_properties(../src/BooleanNetHIP.hip PROPERTIES LANGUAGE HIP)

target_link_libraries(perf_test PUBLIC hip::host)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(perf_test PRIVATE -Wno-deprecated-declarations)
endif()