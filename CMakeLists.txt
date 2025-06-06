cmake_minimum_required(VERSION 3.1...3.31)

set(project_name AlayaLite)
project(${project_name} LANGUAGES CXX)

# options
option(ENABLE_COVERAGE "Enable profiling test coverage" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)

message("Building with CMake version: ${CMAKE_VERSION}")

# please install clang-18 or g++-13
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -march=x86-64 ${CMAKE_CXX_FLAGS_RELEASE}")  
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -march=x86-64 ${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# define third_party
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(libcoro REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
set(THIRD_PARTY_LIBS spdlog::spdlog fmt::fmt concurrentqueue::concurrentqueue libcoro::libcoro)
set(GTEST_LIBS GTest::gtest GTest::gtest_main)

add_definitions(-DSPDLOG_HEADER_ONLY)
add_definitions(-DFMT_HEADER_ONLY)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(${project_name} INTERFACE)
target_link_libraries(${project_name} INTERFACE ${THIRD_PARTY_LIBS})
if (ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()
add_subdirectory(python)
