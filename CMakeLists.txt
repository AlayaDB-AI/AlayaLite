cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(project_name AlayaLite)
project(${project_name} LANGUAGES CXX)

# ============================================================================
# INCLUDE MODULAR CONFIGURATION FILES
# ============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerConfig)     # Project options and validation
include(OptionConfig)       # Project options and validation
include(ConanDepConfig)     # Conan integration and dependency management

# ============================================================================
# PROJECT TARGETS CONFIGURATION
# ============================================================================
message(STATUS "Configuring project targets...")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Define main library target as interface (header-only)
add_library(${project_name} INTERFACE)
target_link_libraries(${project_name} INTERFACE ${THIRD_PARTY_LIBS})

# Set target properties for modern CMake
target_include_directories(${project_name} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Add compile features
target_compile_features(${project_name} INTERFACE cxx_std_20)
message(STATUS "Main target '${project_name}' configured as interface library")

# Include subdirectories based on configuration
if(ENABLE_UNIT_TESTS)
  add_subdirectory(tests)
  message(STATUS "Added tests subdirectory")
endif()

# Always include Python bindings
add_subdirectory(python)
message(STATUS "Added python subdirectory")

message(STATUS "${project_name} configuration completed successfully!")
