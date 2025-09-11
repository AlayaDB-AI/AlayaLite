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


# Explicitly set Visual Studio toolset for Windows (replaces Conan conf)
if(MSVC)
  set(CMAKE_GENERATOR_TOOLSET "v143" CACHE STRING "Visual Studio 2022 toolset" FORCE)
  message(STATUS "Configured VS toolset: ${CMAKE_GENERATOR_TOOLSET}")
endif()

# Execute Conan based on platform with proper command formatting
if(APPLE)
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Macos"
    RESULT_VARIABLE conan_result
  )
  # In MAC, the following line (include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")) should between two execute_process
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOSX_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Reset sysroot: canon may set it to a incorrect value.
  set(CMAKE_OSX_SYSROOT "${MACOSX_SDK_PATH}" CACHE STRING "macOS SDK path" FORCE)
  message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")

  # Architecture support: M1/M2
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture" FORCE)

elseif(WIN32)
  # Force set Conan generators directory to avoid temp path issues
  set(CONAN_GENERATORS_DIR "${PROJECT_SOURCE_DIR}/build/generators")
  message("CONAN_GENERATORS_DIR: ${CONAN_GENERATORS_DIR}")

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

else()
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators
    RESULT_VARIABLE conan_result
  )
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")
endif()

# Check Conan execution result before proceeding
if(NOT conan_result EQUAL 0)
  message(FATAL_ERROR "Conan execution failed with error code: ${conan_result}")
endif()


# C++ standard configuration (matches project requirements)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compilation options with platform-specific adjustments
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Ensure PIC for shared libraries

# Check AVX512 support
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-march=native")
set(AVX512_RUN_CODE "
#include <immintrin.h>
int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    return 0;
}
")

check_cxx_source_runs("${AVX512_RUN_CODE}" AVX512_CAN_RUN)
# RaBitQ optimization
if (AVX512_CAN_RUN)
  message(STATUS "Can use AVX512.")
  add_compile_options(
      -mfma
      -mavx512f
      -mavx512dq
      -mavx512bw
      -mavx512vl
      -mavx512bitalg
      -mavx512vpopcntdq
  )
else()
    message(STATUS "AVX-512 not supported or cannot run.")
endif()

if(MSVC)
  # Windows MSVC specific flags
  message(STATUS "Using MSVC compiler settings")
  set(CMAKE_CXX_FLAGS "/EHsc /utf-8 ${CMAKE_CXX_FLAGS}")  # Exception handling and UTF-8 support
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
else()
  # GCC/Clang flags
  message(STATUS "Using GCC/Clang compiler settings")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS}")  # Enable common warnings
  # set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Set output directory for executable binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define project root directory macro for source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Find dependencies installed by Conan (order matters for linking)
find_package(OpenMP REQUIRED)
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Configure third-party libraries based on operating system
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(libcoro REQUIRED)
  set(THIRD_PARTY_LIBS
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    fmt::fmt
    concurrentqueue::concurrentqueue
    libcoro::libcoro
    Eigen3::Eigen
  )
else()
  set(THIRD_PARTY_LIBS
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    fmt::fmt
    concurrentqueue::concurrentqueue
    Eigen3::Eigen
  )
endif()

if(ENABLE_UNIT_TESTS)
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
