# Conan configuration and dependency management This module handles Conan integration, platform-specific settings, and
# dependency management

message(STATUS "Configuring Conan and dependencies for platform: ${CMAKE_SYSTEM_NAME}")

# ============================================================================
# CONAN PLATFORM CONFIGURATION
# ============================================================================

# Execute Conan based on platform with proper command formatting
if(APPLE)
  message(STATUS "Setting up macOS Conan configuration...")

  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Macos"
            ${CMAKE_BUILD_TYPE} RESULT_VARIABLE conan_result
  )

  # In macOS, the toolchain must be included between execute_process calls
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")

  # Get macOS SDK path
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOSX_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Reset sysroot: Conan may set it to an incorrect value
  set(CMAKE_OSX_SYSROOT
      "${MACOSX_SDK_PATH}"
      CACHE STRING "macOS SDK path" FORCE
  )
  message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")

  # Architecture support: Apple Silicon (M1/M2)
  set(CMAKE_OSX_ARCHITECTURES
      "arm64"
      CACHE STRING "Build architecture" FORCE
  )
  message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

elseif(WIN32)
  message(STATUS "Setting up Windows Conan configuration...")

  # Force set Conan generators directory to avoid temp path issues
  set(CONAN_GENERATORS_DIR "${PROJECT_SOURCE_DIR}/build/generators")
  message(STATUS "Conan generators directory: ${CONAN_GENERATORS_DIR}")

  # Execute Conan with proper Windows command handling
  execute_process(
    COMMAND cmd /c "${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook_win.bat" "${CONAN_GENERATORS_DIR}" "Windows"
            ${CMAKE_BUILD_TYPE}
    RESULT_VARIABLE conan_result
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE conan_output
    ERROR_VARIABLE conan_error
  )

  # Debug output for Windows issues
  if(conan_output)
    message(STATUS "Conan output: ${conan_output}")
  endif()
  if(conan_error)
    message(STATUS "Conan error: ${conan_error}")
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
  message(STATUS "Included Conan toolchain: ${CONAN_TOOLCHAIN_FILE}")

  # Linux/Unix configuration
else()
  message(STATUS "Setting up Linux/Unix Conan configuration...")

  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Linux"
            ${CMAKE_BUILD_TYPE} RESULT_VARIABLE conan_result
  )
  include("${PROJECT_BINARY_DIR}/generators/conan_toolchain.cmake")
endif()

# Check Conan execution result before proceeding
if(NOT conan_result EQUAL 0)
  message(FATAL_ERROR "Conan execution failed with error code: ${conan_result}")
endif()

message(STATUS "Conan platform configuration completed successfully")

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

message(STATUS "Configuring project dependencies...")

# Find common dependencies installed by Conan (order matters for linking)
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(RocksDB REQUIRED)

# OpenMP: only try to find if Conan provided it (i.e., libomp is used) Otherwise, rely on compiler flags and system
# linker (GCC on Linux)
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/OpenMPConfig.cmake" OR EXISTS "${CMAKE_PREFIX_PATH}/OpenMPConfig.cmake")
  find_package(OpenMP REQUIRED)
  set(OPENMP_TARGET OpenMP::OpenMP_CXX)
else()
  # Fallback: assume OpenMP is available via compiler flags (e.g., GCC on Linux) No explicit target; linking handled
  # implicitly
  set(OPENMP_TARGET "")
  message(STATUS "OpenMP: Using compiler-provided OpenMP (no CMake target)")
endif()

# Configure common third-party libraries
set(COMMON_THIRD_PARTY_LIBS spdlog::spdlog concurrentqueue::concurrentqueue Eigen3::Eigen RocksDB::rocksdb)
# Add OpenMP target only if available
if(OPENMP_TARGET)
  list(APPEND COMMON_THIRD_PARTY_LIBS ${OPENMP_TARGET})
endif()

# Add platform-specific libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Adding Linux-specific dependencies...")
  find_package(libcoro REQUIRED)
  find_package(liburing REQUIRED)
  list(APPEND COMMON_THIRD_PARTY_LIBS libcoro::libcoro liburing::liburing)
endif()

# Set final third-party libraries list
set(THIRD_PARTY_LIBS ${COMMON_THIRD_PARTY_LIBS})

message(STATUS "================= CONAN DONE ==================")

# Configure testing framework
if(ENABLE_UNIT_TESTS)
  find_package(GTest REQUIRED)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  message(STATUS "UNIT TEST    : ENABLED - GTest configured")
else()
  message(STATUS "UNIT TEST    : DISABLED - Skipping GTest configuration")
endif()

if(ENABLE_COVERAGE)
  message(STATUS "CODECOV TEST : ENABLED - Coverage flags will be applied to tests")
else()
  message(STATUS "CODECOV TEST : DISABLED")
endif()
