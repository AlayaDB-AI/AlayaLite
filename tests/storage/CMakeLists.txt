
set(STORAGE_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/sequential_storage_test.cpp
)

add_executable(storage_test ${STORAGE_TEST_SRC})

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(storage_test PRIVATE --coverage -O0 -g)
        target_link_libraries(storage_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(storage_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(storage_test PRIVATE -fprofile-instr-generate)
        target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    endif()
else()
    target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
