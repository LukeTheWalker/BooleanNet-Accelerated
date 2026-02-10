cmake_minimum_required(VERSION 3.25)
project("BooleanNet" CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

list(APPEND CMAKE_MODULE_PATH "/opt/rocm/hip/cmake")
find_package(hip REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS "src/*.cpp")
set(HIP_FILES "src/BooleanNetHIP.hip")

add_executable(${PROJECT_NAME} 
    ${CPP_FILES}
    ${HIP_FILES}
)

set_source_files_properties(${HIP_FILES} PROPERTIES LANGUAGE HIP)

target_link_libraries(${PROJECT_NAME} PUBLIC hip::host)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(perf_tests)