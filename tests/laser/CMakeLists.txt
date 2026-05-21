# Gate L1 — mechanical port compile smoke test. Verifies the Laser headers under include/index/graph/laser/ compile
# under AlayaLite's toolchain. See openspec/changes/port-laser-disk-index tasks.md item 2.13.

if(NOT ALAYA_ENABLE_LASER)
  return()
endif()

add_executable(test_laser_compile ${CMAKE_CURRENT_SOURCE_DIR}/test_laser_compile.cpp)
target_link_libraries(test_laser_compile PRIVATE alaya_laser)
target_compile_features(test_laser_compile PRIVATE cxx_std_20)
# Explicit SIMD flags for Laser (see CMakeLists.txt _ALAYA_LASER_CONSUMER_OPTIONS) plus the standard test-only flags.
target_compile_options(
  test_laser_compile
  PRIVATE -O3
          -DNDEBUG
          -Wall
          -Wextra
          ${_ALAYA_LASER_CONSUMER_OPTIONS}
)

add_test(NAME laser_test_compile COMMAND $<TARGET_FILE:test_laser_compile>)

# Round-trip test for FHTRotator::dump_signs: seeded determinism, seed sensitivity, and on-disk layout. See the test
# source for details.
add_executable(test_rotator_dump ${CMAKE_CURRENT_SOURCE_DIR}/test_rotator_dump.cpp)
target_link_libraries(test_rotator_dump PRIVATE alaya_laser)
target_compile_features(test_rotator_dump PRIVATE cxx_std_20)
target_compile_options(
  test_rotator_dump
  PRIVATE -O3
          -DNDEBUG
          -Wall
          -Wextra
          ${_ALAYA_LASER_CONSUMER_OPTIONS}
)

add_test(NAME laser_test_rotator_dump COMMAND $<TARGET_FILE:test_rotator_dump>)

add_executable(test_laser_page_layout_round_trip ${CMAKE_CURRENT_SOURCE_DIR}/qg/test_page_layout_round_trip.cpp)
target_link_libraries(test_laser_page_layout_round_trip PRIVATE alaya_laser ${GTEST_LIBS})
target_compile_features(test_laser_page_layout_round_trip PRIVATE cxx_std_20)
target_compile_definitions(test_laser_page_layout_round_trip PRIVATE ALAYA_LASER_TESTING)
target_compile_options(
  test_laser_page_layout_round_trip
  PRIVATE -O3
          -DNDEBUG
          -Wall
          -Wextra
          ${_ALAYA_LASER_CONSUMER_OPTIONS}
)

add_test(NAME laser_test_page_layout_round_trip COMMAND $<TARGET_FILE:test_laser_page_layout_round_trip>)
set_tests_properties(laser_test_page_layout_round_trip PROPERTIES LABELS laser)

add_executable(laser_simd_dispatch_test ${CMAKE_CURRENT_SOURCE_DIR}/simd_dispatch_test.cpp)
target_link_libraries(laser_simd_dispatch_test PRIVATE alaya_laser ${GTEST_LIBS})
target_compile_features(laser_simd_dispatch_test PRIVATE cxx_std_20)
target_compile_options(laser_simd_dispatch_test PRIVATE -Wall -Wextra ${_ALAYA_LASER_CONSUMER_OPTIONS})

add_test(NAME laser_test_simd_dispatch COMMAND $<TARGET_FILE:laser_simd_dispatch_test>)
set_tests_properties(laser_test_simd_dispatch PROPERTIES LABELS "laser;simd")
