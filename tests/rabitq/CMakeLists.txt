add_executable(rabitq_performance_test ${CMAKE_CURRENT_SOURCE_DIR}/deep1M_test.cpp)
target_link_libraries(rabitq_performance_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_test(NAME rabitq_test_performance_on_deep1M COMMAND $<TARGET_FILE:rabitq_performance_test>)
