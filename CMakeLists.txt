cmake_minimum_required(VERSION 3.25)
project("BooleanNet" CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CUDA_ARCHITECTURES "86")

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS "src/*.cpp")
set(CUDA_FILES "src/BooleanNetCUDA.cu" "src/main.cu")

add_executable(${PROJECT_NAME} 
    ${CPP_FILES}
    ${CUDA_FILES}
)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()

if (CMAKE_CUDA_COMPILER)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(perf_tests)
