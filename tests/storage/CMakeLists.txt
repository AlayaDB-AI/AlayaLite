add_executable(storage_test ${CMAKE_CURRENT_SOURCE_DIR}/sequential_storage_test.cpp)
add_executable(static_storage_test ${CMAKE_CURRENT_SOURCE_DIR}/static_storage_test.cpp)

target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(static_storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

add_test(NAME storage_test_sequential_storage COMMAND $<TARGET_FILE:storage_test>)
add_test(NAME storage_test_static_storage COMMAND $<TARGET_FILE:static_storage_test>)
