add_executable(disk_types_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_types_test.cpp)
add_executable(segment_manifest_test ${CMAKE_CURRENT_SOURCE_DIR}/segment_manifest_test.cpp)
add_executable(mmap_file_test ${CMAKE_CURRENT_SOURCE_DIR}/mmap_file_test.cpp)
add_executable(disk_flat_builder_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_flat_builder_test.cpp)
add_executable(disk_flat_searcher_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_flat_searcher_test.cpp)
add_executable(disk_collection_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_collection_test.cpp)
add_executable(test_disk_collection_lock ${CMAKE_CURRENT_SOURCE_DIR}/test_disk_collection_lock.cpp)
add_executable(segment_factory_test ${CMAKE_CURRENT_SOURCE_DIR}/segment_factory_test.cpp)
add_executable(
  disk_collection_factory_dispatch_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_collection_factory_dispatch_test.cpp
)
add_executable(test_vamana_reader ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_reader.cpp)
add_executable(test_vamana_greedy_search ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_greedy_search.cpp)
add_executable(test_vamana_segment_builder ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_segment_builder.cpp)
add_executable(test_vamana_segment_searcher ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_segment_searcher.cpp)
add_executable(test_segment_factory_vamana ${CMAKE_CURRENT_SOURCE_DIR}/test_segment_factory_vamana.cpp)
add_executable(test_disk_collection_vamana ${CMAKE_CURRENT_SOURCE_DIR}/test_disk_collection_vamana.cpp)
add_executable(test_vamana_adapter_overhead ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_adapter_overhead.cpp)
add_executable(test_disk_collection_batch_search ${CMAKE_CURRENT_SOURCE_DIR}/test_disk_collection_batch_search.cpp)

if(ALAYA_ENABLE_LASER)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )

  set(ALAYA_LASER_SEGMENT_FIXTURE_PREFIX dsqg_seg_00000001)
  set(ALAYA_LASER_SEGMENT_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures/laser_segment)
  set(ALAYA_LASER_SEGMENT_FIXTURE_INPUT
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_input.fbin
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_INDEX
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_R64_MD128.index
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS
      ${ALAYA_LASER_SEGMENT_FIXTURE_INPUT}
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_rotator
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_cache_ids
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_cache_nodes
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca.bin
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids_indices
  )

  add_custom_command(
    OUTPUT ${ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS}
    BYPRODUCTS ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca_base.fbin
               ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_vamana_graph.index
               ${CMAKE_CURRENT_BINARY_DIR}/fixtures/python_stage
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target _alayalitepy
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
    COMMAND
      ${CMAKE_COMMAND} -E env ALAYALITE_LASER_BUILD_DIR=${CMAKE_BINARY_DIR} ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py --output-dir ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
      --prefix ${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX} --count 2048 --dim 128 --R 64 --seed 42 --optional-sidecars
    DEPENDS ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py
    COMMENT "Generating deterministic LASER disk fixture"
    VERBATIM
  )
  add_custom_target(laser_segment_fixture DEPENDS ${ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS})

  set(ALAYA_LASER_SEGMENT_TEST_CANDIDATES
      test_laser_segment_importer.cpp
      test_laser_segment_searcher.cpp
      test_segment_factory_laser.cpp
      test_disk_collection_laser.cpp
      test_laser_adapter_overhead.cpp
      test_laser_searcher_thread_safety.cpp
  )
  set(ALAYA_LASER_SEGMENT_TEST_SOURCES)
  foreach(laser_segment_test_source IN LISTS ALAYA_LASER_SEGMENT_TEST_CANDIDATES)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${laser_segment_test_source})
      list(APPEND ALAYA_LASER_SEGMENT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${laser_segment_test_source})
    endif()
  endforeach()

  if(ALAYA_LASER_SEGMENT_TEST_SOURCES)
    set(ALAYA_LASER_SEGMENT_TEST_TARGETS)
    foreach(laser_segment_test_source IN LISTS ALAYA_LASER_SEGMENT_TEST_SOURCES)
      get_filename_component(_laser_segment_test_target ${laser_segment_test_source} NAME_WE)
      add_executable(${_laser_segment_test_target} ${laser_segment_test_source})
      add_dependencies(${_laser_segment_test_target} laser_segment_fixture)
      target_link_libraries(${_laser_segment_test_target} PRIVATE ${PROJECT_NAME} alaya_laser ${GTEST_LIBS})
      target_compile_options(${_laser_segment_test_target} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
      target_compile_definitions(
        ${_laser_segment_test_target}
        PRIVATE ALAYA_ENABLE_LASER=1 ALAYA_LASER_FIXTURE_DIR="${ALAYA_LASER_SEGMENT_FIXTURE_DIR}"
                ALAYA_LASER_FIXTURE_PREFIX="${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}"
      )
      add_coverage_to_target(${_laser_segment_test_target})
      add_test(NAME ${_laser_segment_test_target} COMMAND $<TARGET_FILE:${_laser_segment_test_target}>)
      list(APPEND ALAYA_LASER_SEGMENT_TEST_TARGETS ${_laser_segment_test_target})
    endforeach()
    add_custom_target(test_laser_segment DEPENDS ${ALAYA_LASER_SEGMENT_TEST_TARGETS})
  endif()

  # The hybrid Flat/Vamana/Laser batch_search test is registered as a standard executable above; only its Laser-engine
  # branch needs the fixture, the alaya_laser link, and ALAYA_ENABLE_LASER=1.
  add_dependencies(test_disk_collection_batch_search laser_segment_fixture)
  target_link_libraries(test_disk_collection_batch_search PRIVATE alaya_laser)
  target_compile_options(test_disk_collection_batch_search PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  target_compile_definitions(
    test_disk_collection_batch_search
    PRIVATE ALAYA_ENABLE_LASER=1 ALAYA_LASER_FIXTURE_DIR="${ALAYA_LASER_SEGMENT_FIXTURE_DIR}"
            ALAYA_LASER_FIXTURE_PREFIX="${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}"
  )
else()
  set(ALAYA_LASER_NEGATIVE_TEST_CANDIDATES test_laser_segment_importer.cpp test_laser_segment_searcher.cpp
                                           test_segment_factory_laser.cpp test_disk_collection_laser.cpp
  )
  set(ALAYA_LASER_NEGATIVE_TEST_TARGETS)
  foreach(laser_negative_test_source IN LISTS ALAYA_LASER_NEGATIVE_TEST_CANDIDATES)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${laser_negative_test_source})
      get_filename_component(_laser_negative_test_stem ${laser_negative_test_source} NAME_WE)
      set(_laser_negative_test_target ${_laser_negative_test_stem}_negative)
      add_executable(${_laser_negative_test_target} ${CMAKE_CURRENT_SOURCE_DIR}/${laser_negative_test_source})
      target_link_libraries(${_laser_negative_test_target} PRIVATE ${PROJECT_NAME} ${GTEST_LIBS})
      add_coverage_to_target(${_laser_negative_test_target})
      add_test(NAME ${_laser_negative_test_target} COMMAND $<TARGET_FILE:${_laser_negative_test_target}>)
      list(APPEND ALAYA_LASER_NEGATIVE_TEST_TARGETS ${_laser_negative_test_target})
    endif()
  endforeach()
  if(ALAYA_LASER_NEGATIVE_TEST_TARGETS)
    add_custom_target(test_laser_segment_negative DEPENDS ${ALAYA_LASER_NEGATIVE_TEST_TARGETS})
  endif()
endif()

target_link_libraries(disk_types_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(segment_manifest_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(mmap_file_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(disk_flat_builder_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(disk_flat_searcher_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(disk_collection_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_disk_collection_lock PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(segment_factory_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(disk_collection_factory_dispatch_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_vamana_reader PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_vamana_greedy_search PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_vamana_segment_builder PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_vamana_segment_searcher PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_segment_factory_vamana PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_disk_collection_vamana PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_vamana_adapter_overhead PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(test_disk_collection_batch_search PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_coverage_to_target(disk_types_test)
add_coverage_to_target(segment_manifest_test)
add_coverage_to_target(mmap_file_test)
add_coverage_to_target(disk_flat_builder_test)
add_coverage_to_target(disk_flat_searcher_test)
add_coverage_to_target(disk_collection_test)
add_coverage_to_target(test_disk_collection_lock)
add_coverage_to_target(segment_factory_test)
add_coverage_to_target(disk_collection_factory_dispatch_test)
add_coverage_to_target(test_vamana_reader)
add_coverage_to_target(test_vamana_greedy_search)
add_coverage_to_target(test_vamana_segment_builder)
add_coverage_to_target(test_vamana_segment_searcher)
add_coverage_to_target(test_segment_factory_vamana)
add_coverage_to_target(test_disk_collection_vamana)
add_coverage_to_target(test_vamana_adapter_overhead)
add_coverage_to_target(test_disk_collection_batch_search)

add_test(NAME disk_test_types COMMAND $<TARGET_FILE:disk_types_test>)
add_test(NAME disk_test_segment_manifest COMMAND $<TARGET_FILE:segment_manifest_test>)
add_test(NAME disk_test_mmap_file COMMAND $<TARGET_FILE:mmap_file_test>)
add_test(NAME disk_test_flat_builder COMMAND $<TARGET_FILE:disk_flat_builder_test>)
add_test(NAME disk_test_flat_searcher COMMAND $<TARGET_FILE:disk_flat_searcher_test>)
add_test(NAME disk_test_collection COMMAND $<TARGET_FILE:disk_collection_test>)
add_test(NAME test_disk_collection_lock COMMAND $<TARGET_FILE:test_disk_collection_lock>)
add_test(NAME disk_test_segment_factory COMMAND $<TARGET_FILE:segment_factory_test>)
add_test(NAME disk_test_collection_factory_dispatch COMMAND $<TARGET_FILE:disk_collection_factory_dispatch_test>)
add_test(NAME test_vamana_reader COMMAND $<TARGET_FILE:test_vamana_reader>)
add_test(NAME test_vamana_greedy_search COMMAND $<TARGET_FILE:test_vamana_greedy_search>)
add_test(NAME test_vamana_segment_builder COMMAND $<TARGET_FILE:test_vamana_segment_builder>)
add_test(NAME test_vamana_segment_searcher COMMAND $<TARGET_FILE:test_vamana_segment_searcher>)
add_test(NAME test_segment_factory_vamana COMMAND $<TARGET_FILE:test_segment_factory_vamana>)
add_test(NAME test_disk_collection_vamana COMMAND $<TARGET_FILE:test_disk_collection_vamana>)
add_test(NAME test_vamana_adapter_overhead COMMAND $<TARGET_FILE:test_vamana_adapter_overhead>)
add_test(NAME test_disk_collection_batch_search COMMAND $<TARGET_FILE:test_disk_collection_batch_search>)
