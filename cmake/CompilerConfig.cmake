# Compiler configuration and flags This module configures C++ standard, compiler flags, and build settings

# C++ standard configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure position-independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform-specific compiler flags
if(MSVC)
  # Windows MSVC specific flags
  message(STATUS "Configuring for MSVC compiler")
  # Exception handling and UTF-8 support
  set(CMAKE_CXX_FLAGS "/EHsc /utf-8 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")

  # Set Visual Studio toolset
  set(CMAKE_GENERATOR_TOOLSET
      "v143"
      CACHE STRING "Visual Studio 2022 toolset" FORCE
  )
  message(STATUS "Configured VS toolset: ${CMAKE_GENERATOR_TOOLSET}")
else()
  # GCC/Clang flags
  message(STATUS "Configuring for GCC/Clang compiler")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS}") # Enable common
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define project root directory macro for source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")
