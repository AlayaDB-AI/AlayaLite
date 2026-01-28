function(add_coverage_to_target target_name)
  if(NOT ENABLE_COVERAGE)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang
    target_compile_options(${target_name} PRIVATE --coverage -O0 -g)
    target_link_libraries(${target_name} PRIVATE --coverage)
    message(STATUS "Coverage enabled for ${target_name} (GCC/Clang)")

  elseif(MSVC)
    # MSVC
    target_compile_options(${target_name} PRIVATE /Z7)
    target_link_options(${target_name} PRIVATE /DEBUG)
    message(STATUS "Coverage enabled for ${target_name} (MSVC/OpenCppCoverage)")
  endif()
endfunction()
