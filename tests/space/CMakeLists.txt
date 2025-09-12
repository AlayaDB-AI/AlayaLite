set(SPACE_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/quant/sq4_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant/sq8_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/raw_space_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sq4_space_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sq8_space_test.cpp
)

add_executable(space_test ${SPACE_TEST_SRC})

target_link_libraries(space_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

# Add test to CTest
add_test(NAME SpaceTest COMMAND space_test)
