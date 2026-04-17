add_executable(query_utils_test ${CMAKE_CURRENT_SOURCE_DIR}/query_utils_test.cpp)
add_executable(log_test ${CMAKE_CURRENT_SOURCE_DIR}/log_test.cpp)
add_executable(evaluate_test ${CMAKE_CURRENT_SOURCE_DIR}/evaluate_test.cpp)
add_executable(metric_type_test ${CMAKE_CURRENT_SOURCE_DIR}/metric_type_test.cpp)
add_executable(data_utils_test ${CMAKE_CURRENT_SOURCE_DIR}/data_utils_test.cpp)
add_executable(dataset_utils_test ${CMAKE_CURRENT_SOURCE_DIR}/dataset_utils_test.cpp)
add_executable(index_encoding_test ${CMAKE_CURRENT_SOURCE_DIR}/index_encoding_test.cpp)
add_executable(metadata_filter_test ${CMAKE_CURRENT_SOURCE_DIR}/metadata_filter_test.cpp)
add_executable(rotator_utils_test ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_utils/rotator_test.cpp)
add_executable(lut_utils_test ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_utils/lut_test.cpp)
add_executable(math_test ${CMAKE_CURRENT_SOURCE_DIR}/math_test.cpp)

target_link_libraries(query_utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(log_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(evaluate_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(metric_type_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(data_utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(dataset_utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(index_encoding_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(metadata_filter_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(rotator_utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(lut_utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(math_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_coverage_to_target(query_utils_test)
add_coverage_to_target(log_test)
add_coverage_to_target(evaluate_test)
add_coverage_to_target(metric_type_test)
add_coverage_to_target(data_utils_test)
add_coverage_to_target(dataset_utils_test)
add_coverage_to_target(index_encoding_test)
add_coverage_to_target(metadata_filter_test)
add_coverage_to_target(rotator_utils_test)
add_coverage_to_target(lut_utils_test)
add_coverage_to_target(math_test)

add_test(NAME utils_test_query_utils COMMAND $<TARGET_FILE:query_utils_test>)
add_test(NAME utils_test_log COMMAND $<TARGET_FILE:log_test>)
add_test(NAME utils_test_evaluate COMMAND $<TARGET_FILE:evaluate_test>)
add_test(NAME utils_test_metric_type COMMAND $<TARGET_FILE:metric_type_test>)
add_test(NAME utils_test_data_utils COMMAND $<TARGET_FILE:data_utils_test>)
add_test(NAME utils_test_dataset_utils COMMAND $<TARGET_FILE:dataset_utils_test>)
add_test(NAME utils_test_index_encoding COMMAND $<TARGET_FILE:index_encoding_test>)
add_test(NAME utils_test_metadata_filter COMMAND $<TARGET_FILE:metadata_filter_test>)
add_test(NAME utils_test_rotator_utils COMMAND $<TARGET_FILE:rotator_utils_test>)
add_test(NAME utils_test_lut_utils COMMAND $<TARGET_FILE:lut_utils_test>)
add_test(NAME utils_test_math COMMAND $<TARGET_FILE:math_test>)
set_tests_properties(
  utils_test_query_utils
  utils_test_log
  utils_test_evaluate
  utils_test_metric_type
  utils_test_data_utils
  utils_test_dataset_utils
  utils_test_index_encoding
  utils_test_metadata_filter
  utils_test_rotator_utils
  utils_test_lut_utils
  utils_test_math
  PROPERTIES LABELS "utils"
)
