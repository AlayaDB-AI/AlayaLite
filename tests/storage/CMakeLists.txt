add_executable(storage_test ${CMAKE_CURRENT_SOURCE_DIR}/sequential_storage_test.cpp)

target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

add_test(NAME storage_test_sequential_storage COMMAND $<TARGET_FILE:storage_test>)
