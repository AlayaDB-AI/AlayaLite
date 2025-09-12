set(STORAGE_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sequential_storage_test.cpp)

add_executable(storage_test ${STORAGE_TEST_SRC})

target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

# Add test to CTest
add_test(NAME StorageTest COMMAND storage_test)
