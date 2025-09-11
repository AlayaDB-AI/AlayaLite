# Conan configuration and dependency management
# This module handles Conan integration, platform-specific settings, and dependency management

message(STATUS "Configuring Conan and dependencies for platform: ${CMAKE_SYSTEM_NAME}")

# ============================================================================
# CONAN PLATFORM CONFIGURATION
# ============================================================================

# Execute Conan based on platform with proper command formatting
if(APPLE)
  message(STATUS "Setting up macOS Conan configuration...")

  execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators "Macos"
    RESULT_VARIABLE conan_result
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
  set(CMAKE_OSX_SYSROOT "${MACOSX_SDK_PATH}" CACHE STRING "macOS SDK path" FORCE)
  message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")

  # Architecture support: Apple Silicon (M1/M2)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture" FORCE)
  message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

elseif(WIN32)
  message(STATUS "Setting up Windows Conan configuration...")

  # Force set Conan generators directory to avoid temp path issues
  set(CONAN_GENERATORS_DIR "${PROJECT_SOURCE_DIR}/build/generators")
  message(STATUS "Conan generators directory: ${CONAN_GENERATORS_DIR}")

  # Execute Conan with proper Windows command handling
  execute_process(
    COMMAND cmd /c "${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook_win.bat" "${CONAN_GENERATORS_DIR}" "Windows"
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
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/conan_build/conan_hook.sh ${PROJECT_BINARY_DIR}/generators
    RESULT_VARIABLE conan_result
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
find_package(fmt REQUIRED)

# Configure common third-party libraries
set(COMMON_THIRD_PARTY_LIBS
  spdlog::spdlog
  fmt::fmt
  concurrentqueue::concurrentqueue
)

# Add platform-specific libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Adding Linux-specific dependencies...")
  find_package(libcoro REQUIRED)
  list(APPEND COMMON_THIRD_PARTY_LIBS libcoro::libcoro)
endif()

# Set final third-party libraries list
set(THIRD_PARTY_LIBS ${COMMON_THIRD_PARTY_LIBS})

# Configure testing framework
if(ENABLE_UNIT_TESTS)
  message(STATUS "Configuring testing framework...")
  find_package(GTest REQUIRED)
  set(GTEST_LIBS
    GTest::gtest
    GTest::gtest_main
  )
  message(STATUS "Unit tests: ENABLED - GTest configured")
else()
  message(STATUS "Unit tests: DISABLED - Skipping GTest configuration")
endif()

if (ENABLE_COVERAGE)
  message(STATUS "Code coverage: ENABLED - Coverage flags will be applied to tests")
else()
  message(STATUS "Code coverage: DISABLED")
endif()

# Print dependency summary
message(STATUS "=========CONAN & DEPENDENCY SUMMARY===========")
message(STATUS "Third-party libraries: ${THIRD_PARTY_LIBS}")
if(ENABLE_UNIT_TESTS)
  message(STATUS "Test libraries: ${GTEST_LIBS}")
endif()
message(STATUS "=============================================")

message(STATUS "Conan configuration and dependency management completed successfully")
