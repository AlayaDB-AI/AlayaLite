# DiskANN disk index tests.
#
# Layout / PQ / node-cache are header-only over std + SIMD (SIMD flags are applied globally by the top-level
# CMakeLists), so they link only the shared third-party libs. Beam search / index / e2e use the LASER AlignedFileReader
# (libaio on Linux) and therefore need the alaya_laser interface, its consumer compile options, and ALAYA_ENABLE_LASER.

set(_diskann_plain_tests test_diskann_layout test_diskann_pq test_diskann_node_cache test_diskann_tombstone_slot)
foreach(test_target IN LISTS _diskann_plain_tests)
  add_executable(${test_target} ${CMAKE_CURRENT_SOURCE_DIR}/${test_target}.cpp)
  target_link_libraries(${test_target} PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
  add_coverage_to_target(${test_target})
  add_test(NAME ${test_target} COMMAND $<TARGET_FILE:${test_target}>)
  set_tests_properties(${test_target} PROPERTIES LABELS "diskann")
endforeach()

if(ALAYA_ENABLE_LASER)
  set(_diskann_laser_tests
      test_diskann_beam_search
      test_diskann_index
      test_diskann_e2e
      test_diskann_disk_page_io
      test_diskann_tombstone_search
      test_diskann_update_e2e
  )
  foreach(test_target IN LISTS _diskann_laser_tests)
    add_executable(${test_target} ${CMAKE_CURRENT_SOURCE_DIR}/${test_target}.cpp)
    target_link_libraries(${test_target} PRIVATE ${PROJECT_NAME} alaya_laser ${GTEST_LIBS})
    target_compile_options(${test_target} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
    target_compile_definitions(${test_target} PRIVATE ALAYA_ENABLE_LASER=1)
    add_coverage_to_target(${test_target})
    add_test(NAME ${test_target} COMMAND $<TARGET_FILE:${test_target}>)
    set_tests_properties(${test_target} PROPERTIES LABELS "diskann")
  endforeach()
  # The end-to-end build+search test is heavier; give it a generous timeout.
  set_tests_properties(test_diskann_e2e PROPERTIES TIMEOUT 600 LABELS "diskann;e2e")

  # Standalone SIFT1M benchmark (manual; NOT registered as a ctest). Measures recall / latency / disk-reads-per-query so
  # the index can be compared against the official DiskANN `search_disk_index` tool on the same data and metrics. Run it
  # explicitly: build target `bench_diskann_sift`, then execute the binary.
  add_executable(bench_diskann_sift ${CMAKE_CURRENT_SOURCE_DIR}/bench_diskann_sift.cpp)
  target_link_libraries(bench_diskann_sift PRIVATE ${PROJECT_NAME} alaya_laser)
  target_compile_options(bench_diskann_sift PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  target_compile_definitions(bench_diskann_sift PRIVATE ALAYA_ENABLE_LASER=1)

endif()
