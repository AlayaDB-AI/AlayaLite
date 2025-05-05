cmake_minimum_required(VERSION 3.1...3.31)

project(AlayaLite
        VERSION 1.0
        LANGUAGES CXX)

# please install clang-18 or g++-13
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=armv8-a -fPIC ")  
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -march=armv8-a -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")
# add depenedencies
add_subdirectory(third_party)

# export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# define third_party
if (NO_COROUTINE)
        set(THIRD_PARTY_LIBS spdlog::spdlog fmt::fmt)
else()
        set(THIRD_PARTY_LIBS spdlog::spdlog fmt::fmt libcoro)
endif()

set(GTEST_LIBS gtest gtest_main)

# options
option(ENABLE_COVERAGE "Enable profiling test coverage" OFF)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# include concurrentqueue
include_directories(${concurrentqueue_SOURCE_DIR})
include_directories(${spdlog_SOURCE_DIR}/include)

add_library(AlayaLite INTERFACE)
target_link_libraries(AlayaLite INTERFACE ${THIRD_PARTY_LIBS})

# add subdirectories
# add_subdirectory(tests)
add_subdirectory(pyalaya)
