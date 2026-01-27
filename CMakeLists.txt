cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(PROJECT_NAME AlayaLite)
project(${PROJECT_NAME} LANGUAGES CXX)

# Include utility functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PrintSummary)

# ============================================================================
# C++ STANDARD AND COMPILER CONFIGURATION
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  message(STATUS "Configuring for MSVC compiler")
  set(CMAKE_CXX_FLAGS "/openmp /EHsc /utf-8 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
  set(CMAKE_GENERATOR_TOOLSET
      "v143"
      CACHE STRING "Visual Studio 2022 toolset" FORCE
  )
else()
  message(STATUS "Configuring for GCC/Clang compiler")
  set(CMAKE_CXX_FLAGS "-fopenmp -Wall ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# ============================================================================
# PROJECT OPTIONS
# ============================================================================

# Testing mode: OFF, ON, or COVERAGE - OFF:      No tests built - ON:       Build and run tests - COVERAGE: Build tests
# with coverage instrumentation (implies Debug build)
set(BUILD_TESTING
    "OFF"
    CACHE STRING "Build testing: OFF, ON, or COVERAGE"
)
set_property(CACHE BUILD_TESTING PROPERTY STRINGS "OFF" "ON" "COVERAGE")

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
  message(STATUS "Build type not specified, defaulting to Release")
endif()

# Configure based on BUILD_TESTING value
if(BUILD_TESTING STREQUAL "COVERAGE")
  set(ENABLE_TESTS ON)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  endif()
elseif(BUILD_TESTING STREQUAL "ON")
  set(ENABLE_TESTS ON)
else()
  set(ENABLE_TESTS OFF)
endif()

# ============================================================================
# CONAN AND DEPENDENCY MANAGEMENT
# ============================================================================

message(STATUS "Configuring Conan and dependencies for platform: ${CMAKE_SYSTEM_NAME}")

if(APPLE)
  message(STATUS "Setting up macOS Conan configuration...")
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Macos"
            ${CMAKE_BUILD_TYPE} RESULT_VARIABLE conan_result
  )
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")

  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOSX_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CMAKE_OSX_SYSROOT
      "${MACOSX_SDK_PATH}"
      CACHE STRING "macOS SDK path" FORCE
  )
  set(CMAKE_OSX_ARCHITECTURES
      "arm64"
      CACHE STRING "Build architecture" FORCE
  )
  message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")

elseif(WIN32)
  message(STATUS "Setting up Windows Conan configuration...")
  set(CONAN_GENERATORS_DIR "${PROJECT_SOURCE_DIR}/build/generators")

  execute_process(
    COMMAND cmd /c "${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook_win.bat" "${CONAN_GENERATORS_DIR}" "Windows"
            ${CMAKE_BUILD_TYPE}
    RESULT_VARIABLE conan_result
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE conan_output
    ERROR_VARIABLE conan_error
  )

  if(conan_output)
    message(STATUS "Conan output: ${conan_output}")
  endif()
  if(conan_error)
    message(STATUS "Conan error: ${conan_error}")
  endif()

  set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")
  if(NOT EXISTS ${CONAN_TOOLCHAIN_FILE})
    set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/generators/conan_toolchain.cmake")
  endif()
  if(NOT EXISTS ${CONAN_TOOLCHAIN_FILE})
    message(FATAL_ERROR "Conan toolchain file not found: ${CONAN_TOOLCHAIN_FILE}")
  endif()
  include(${CONAN_TOOLCHAIN_FILE})

else()
  message(STATUS "Setting up Linux/Unix Conan configuration...")
  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Linux"
            ${CMAKE_BUILD_TYPE} RESULT_VARIABLE conan_result
  )
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")
endif()

if(NOT conan_result EQUAL 0)
  message(FATAL_ERROR "Conan execution failed with error code: ${conan_result}")
endif()

# Find dependencies
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP configuration
set(OPENMP_TARGET "")
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/OpenMPConfig.cmake" OR EXISTS "${CMAKE_PREFIX_PATH}/OpenMPConfig.cmake")
  find_package(OpenMP REQUIRED)
  set(OPENMP_TARGET OpenMP::OpenMP_CXX)
else()
  set(OPENMP_TARGET "")
  message(STATUS "OpenMP: Using compiler-provided OpenMP (no CMake target)")
endif()

set(THIRD_PARTY_LIBS spdlog::spdlog concurrentqueue::concurrentqueue Eigen3::Eigen)
if(OPENMP_TARGET)
  list(APPEND THIRD_PARTY_LIBS ${OPENMP_TARGET})
endif()

# Platform-specific libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(libcoro REQUIRED)
  list(APPEND THIRD_PARTY_LIBS libcoro::libcoro)
endif()

# Testing framework
if(ENABLE_TESTS)
  find_package(GTest REQUIRED)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
endif()

# ============================================================================
# PROJECT TARGETS
# ============================================================================

message(STATUS "Configuring project targets...")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE ${THIRD_PARTY_LIBS})
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# ============================================================================
# SUBDIRECTORIES
# ============================================================================

if(ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

add_subdirectory(python)

print_build_summary()

message(STATUS "${PROJECT_NAME} configuration completed successfully!")
