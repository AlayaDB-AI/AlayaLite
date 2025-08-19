cmake_minimum_required(VERSION 3.15...3.31)

set(project_name AlayaLite)
project(${project_name} LANGUAGES CXX)

# options
option(ENABLE_COVERAGE "Enable profiling test coverage" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)

# set CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Build type not specified, defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

# conan hook
message("asdfasdf${PROJECT_SOURCE_DIR}asdfasdf")
message("asdfasdf${PROJECT_BINARY_DIR}asdfasdf")
execute_process(COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators)
include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")

# please install clang-18 or g++-13
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
set(CMAKE_CXX_FLAGS "-mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bitalg -mavx512vpopcntdq -fopenmp ${CMAKE_CXX_FLAGS}") # rbq optimization
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
add_compile_options(${CONAN_USER_MARCH_FLAGS})

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
find_package(Eigen3 REQUIRED NO_MODULE)
set(THIRD_PARTY_LIBS spdlog::spdlog fmt::fmt concurrentqueue::concurrentqueue libcoro::libcoro Eigen3::Eigen)
set(GTEST_LIBS GTest::gtest GTest::gtest_main)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(${project_name} INTERFACE)
target_link_libraries(${project_name} INTERFACE ${THIRD_PARTY_LIBS})
if (ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()
add_subdirectory(python)
