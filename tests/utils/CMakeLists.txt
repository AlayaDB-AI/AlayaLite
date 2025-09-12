set(UTILS_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/query_utils_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/evaluate_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metric_type_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_utils_test.cpp
)

add_executable(utils_test ${UTILS_TEST_SRC})

target_link_libraries(utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

# Add test to CTest
add_test(NAME UtilsTest COMMAND utils_test)
