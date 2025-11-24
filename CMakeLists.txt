cmake_minimum_required(VERSION 3.25)
project("BooleanNet" CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")

# Enable CUDA
set(alpaka_DEP_CUDA ON)
set(CMAKE_CUDA_ARCHITECTURES "86")

# Set accelerator execution options
set(alpaka_EXEC_CpuSerial OFF)
set(alpaka_DEP_OMP OFF)

add_subdirectory("../common/alpaka3" "${CMAKE_BINARY_DIR}/alpaka_build")

find_package(alpaka REQUIRED)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Automatically gather all .cpp files under src/
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp")

# Create executable with the gathered files
add_executable(${PROJECT_NAME} 
    ${SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)
alpaka_finalize(${PROJECT_NAME})

# Disable deprecated declarations warning
if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()

if (CMAKE_CUDA_COMPILER)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
endif()

# Set include directories using modern CMake
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ../common/alpaka3/include
)
