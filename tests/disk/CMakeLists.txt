add_executable(disk_types_test ${CMAKE_CURRENT_SOURCE_DIR}/disk_types_test.cpp)
add_executable(segment_manifest_test ${CMAKE_CURRENT_SOURCE_DIR}/segment_manifest_test.cpp)

target_link_libraries(disk_types_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(segment_manifest_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_coverage_to_target(disk_types_test)
add_coverage_to_target(segment_manifest_test)

add_test(NAME disk_test_types COMMAND $<TARGET_FILE:disk_types_test>)
add_test(NAME disk_test_segment_manifest COMMAND $<TARGET_FILE:segment_manifest_test>)
