cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(PROJECT_NAME AlayaLite)
project(${PROJECT_NAME} LANGUAGES CXX)

# Include utility functions
include(cmake/PrintSummary.cmake)
include(cmake/Codecov.cmake)

option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(BUILD_TESTING "Build tests" OFF)

# ============================================================================
# 1. C++ STANDARD AND COMPILER CONFIGURATION
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/EHsc /utf-8 /O2)
else()
  add_compile_options(-Wall -Ofast)
endif()

# ============================================================================
# 1. Dependency Management (Auto Conan Setup)
# ============================================================================
include(cmake/ConanSetup.cmake)

find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(THIRD_PARTY_LIBS spdlog::spdlog_header_only concurrentqueue::concurrentqueue Eigen3::Eigen OpenMP::OpenMP_CXX)

if(UNIX AND NOT APPLE)
  find_package(libcoro REQUIRED)
  list(APPEND THIRD_PARTY_LIBS libcoro::libcoro)
endif()

# ============================================================================
# PROJECT TARGETS
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} INTERFACE ${THIRD_PARTY_LIBS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# ============================================================================
# SUBDIRECTORIES
# ============================================================================
if(ENABLE_COVERAGE)
  # Create an interface library named coverage_config
  add_library(coverage_config INTERFACE)

  # Add coverage flags for GCC/Clang
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(coverage_config INTERFACE --coverage -O0 -g)
    target_link_libraries(coverage_config INTERFACE --coverage)
  endif()
endif()
if(BUILD_TESTING)
  include(CTest)
  find_package(GTest)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  add_subdirectory(tests)
endif()

add_subdirectory(python)
print_build_summary()
