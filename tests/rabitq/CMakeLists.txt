add_executable(siftsmall_rabitq_test ${CMAKE_CURRENT_SOURCE_DIR}/siftsmall_test.cpp)
target_link_libraries(siftsmall_rabitq_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_test(NAME rabitq_test_on_siftsmall COMMAND $<TARGET_FILE:siftsmall_rabitq_test>)
