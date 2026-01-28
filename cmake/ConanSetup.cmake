# ConanSetup.cmake - Automatic Conan dependency management This module detects the platform, selects the appropriate
# Conan profile, and runs conan install if needed.

# Skip if already configured (for subsequent cmake runs)
if(_CONAN_SETUP_DONE)
  return()
endif()

# ============================================================================
# Configuration
# ============================================================================
set(CONAN_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts/conan_build")

# cmake_layout generates files to build/<build_type>/generators/
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()
set(CONAN_GENERATORS_DIR "${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/generators")

# ============================================================================
# Detect Platform and Select Profile
# ============================================================================
function(conan_get_profile profile_path_out)
  set(profile_base "${CONAN_SCRIPTS_DIR}")

  if(WIN32)
    set(profile_file "conan_profile_win.x86_64")
  elseif(APPLE)
    set(profile_file "conan_profile_mac.aarch64")
  elseif(UNIX)
    # Detect architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      set(profile_file "conan_profile.aarch64")
    else()
      set(profile_file "conan_profile.x86_64")
    endif()
  else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
  endif()

  set(${profile_path_out}
      "${profile_base}/${profile_file}"
      PARENT_SCOPE
  )
endfunction()

# ============================================================================
# Run Conan Install
# ============================================================================
function(conan_install)
  # Get the appropriate profile
  conan_get_profile(conan_profile)

  if(NOT EXISTS "${conan_profile}")
    message(FATAL_ERROR "Conan profile not found: ${conan_profile}")
  endif()

  message(STATUS "Using Conan profile: ${conan_profile}")

  # Check if conan toolchain already exists (cmake_layout path)
  set(conan_toolchain "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")

  if(EXISTS "${conan_toolchain}")
    message(STATUS "Conan toolchain already exists, skipping conan install")
    return()
  endif()

  # Find conan executable
  find_program(CONAN_EXECUTABLE conan)
  if(NOT CONAN_EXECUTABLE)
    message(FATAL_ERROR "Conan not found! Please install conan: pip install conan")
  endif()

  message(STATUS "Running conan install...")

  # Detect default profile first (ignore errors if already exists)
  execute_process(COMMAND ${CONAN_EXECUTABLE} profile detect --force OUTPUT_QUIET ERROR_QUIET)

  # Run conan install - let cmake_layout handle the output directory
  execute_process(
    COMMAND ${CONAN_EXECUTABLE} install "${CMAKE_SOURCE_DIR}" -pr:h "${conan_profile}" -pr:b default --build=missing
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE conan_result
    OUTPUT_VARIABLE conan_output
    ERROR_VARIABLE conan_error
  )

  if(NOT conan_result EQUAL 0)
    message(FATAL_ERROR "Conan install failed!\n${conan_output}\n${conan_error}")
  endif()

  message(STATUS "Conan install completed successfully")
endfunction()

# ============================================================================
# Main Logic
# ============================================================================

# Run conan install
conan_install()

# Include the generated toolchain from cmake_layout path
set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")
if(EXISTS "${CONAN_TOOLCHAIN_FILE}")
  include("${CONAN_TOOLCHAIN_FILE}")
  message(STATUS "Included Conan toolchain: ${CONAN_TOOLCHAIN_FILE}")
else()
  message(FATAL_ERROR "Conan toolchain not found after install: ${CONAN_TOOLCHAIN_FILE}\n"
                      "Expected location based on cmake_layout: ${CONAN_GENERATORS_DIR}"
  )
endif()

# Mark as done to avoid re-running
set(_CONAN_SETUP_DONE
    TRUE
    CACHE INTERNAL "Conan setup completed"
)
