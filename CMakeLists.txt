cmake_minimum_required(VERSION 3.15...3.31)

set(project_name AlayaLite)
project(${project_name} LANGUAGES CXX)

# Configuration options
option(ENABLE_COVERAGE "Enable profiling test coverage" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Print path information for debugging purposes
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

# Force set Conan generators directory to avoid temp path issues
set(CONAN_GENERATORS_DIR "${PROJECT_SOURCE_DIR}/build/generators")
message("CONAN_GENERATORS_DIR: ${CONAN_GENERATORS_DIR}")

# Explicitly set Visual Studio toolset for Windows (replaces Conan conf)
if(MSVC)
  set(CMAKE_GENERATOR_TOOLSET "v143" CACHE STRING "Visual Studio 2022 toolset" FORCE)
  message(STATUS "Configured VS toolset: ${CMAKE_GENERATOR_TOOLSET}")
endif()

# Execute Conan based on platform with proper command formatting
if(APPLE)
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${CONAN_GENERATORS_DIR} "Macos"
    RESULT_VARIABLE conan_result
  )
elseif(WIN32)
  # Critical: Use absolute path and ensure proper command separation
  execute_process(
    COMMAND cmd /c "${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook_win.bat" "${CONAN_GENERATORS_DIR}" "Windows"
    RESULT_VARIABLE conan_result
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE conan_output
    ERROR_VARIABLE conan_error
  )
  # Print Conan output for debugging Windows issues
  message("Conan output: ${conan_output}")
  message("Conan error: ${conan_error}")
else()
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${CONAN_GENERATORS_DIR}
    RESULT_VARIABLE conan_result
  )
endif()

# Check Conan execution result before proceeding
if(NOT conan_result EQUAL 0)
  message(FATAL_ERROR "Conan execution failed with error code: ${conan_result}")
endif()

# Locate and include Conan toolchain file with fallback path check
set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")
# Fallback check for common Conan output structure
if(NOT EXISTS ${CONAN_TOOLCHAIN_FILE})
  set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/generators/conan_toolchain.cmake")
endif()
# Final validation of toolchain file existence
if(NOT EXISTS ${CONAN_TOOLCHAIN_FILE})
  message(FATAL_ERROR "Conan toolchain file not found: ${CONAN_TOOLCHAIN_FILE}")
endif()
include(${CONAN_TOOLCHAIN_FILE})

# C++ standard configuration (matches project requirements)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compilation options with platform-specific adjustments
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Ensure PIC for shared libraries
if(MSVC)
  # Windows MSVC specific flags
  set(CMAKE_CXX_FLAGS "/EHsc /utf-8 ${CMAKE_CXX_FLAGS}")  # Exception handling and UTF-8 support
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
else()
  # GCC/Clang flags
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS}")  # Enable common warnings
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Set output directory for executable binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define project root directory macro for source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Find dependencies installed by Conan (order matters for linking)
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

# Configure third-party libraries based on operating system
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(libcoro REQUIRED)
  set(THIRD_PARTY_LIBS 
    spdlog::spdlog 
    fmt::fmt 
    concurrentqueue::concurrentqueue 
    libcoro::libcoro
  )
else()
  set(THIRD_PARTY_LIBS 
    spdlog::spdlog 
    fmt::fmt 
    concurrentqueue::concurrentqueue
  )
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
endif()

# Include project header directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define main library target as interface (header-only)
add_library(${project_name} INTERFACE)
target_link_libraries(${project_name} INTERFACE ${THIRD_PARTY_LIBS})

# Include unit tests if enabled
if(ENABLE_UNIT_TESTS)
  add_subdirectory(tests)
endif()

# Include Python bindings subdirectory
add_subdirectory(python)