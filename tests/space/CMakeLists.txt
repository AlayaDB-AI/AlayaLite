
set(SPACE_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/quant/sq4_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant/sq8_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_space_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sq4_space_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sq8_space_test.cpp
)


add_executable(space_test ${SPACE_TEST_SRC})

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(space_test PRIVATE --coverage -O0 -g)
        target_link_libraries(space_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(space_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(space_test PRIVATE -fprofile-instr-generate)
        target_link_libraries(space_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    endif()
else()
    target_link_libraries(space_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
