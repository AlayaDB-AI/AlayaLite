# ============================================================================
# tools/build_vamana_index — standalone C++ executable that depends on
# AlayaLite's headers.
# ============================================================================
#
# build_vamana_index — port of DiskANN's in-memory Vamana builder, producing
# a .index file in DiskANN's single-file binary layout for consumption by any
# downstream DiskANN-format reader.

add_executable(build_vamana_index build_vamana_for_laser.cpp)
target_link_libraries(build_vamana_index PRIVATE ${PROJECT_NAME})
target_compile_features(build_vamana_index PRIVATE cxx_std_20)

# Performance-oriented flags for the research tool. Spec task 3.5 calls for
# `-O3 -march=native -DNDEBUG`; the Release config already sets -O3/-DNDEBUG
# (top-level appends -Ofast which subsumes -O3), so we only need -march=native
# here to enable the host's best ISA beyond what distance_l2.ipp's per-function
# `ALAYA_TARGET_AVX*` attributes already unlock.
if(NOT MSVC)
  target_compile_options(build_vamana_index PRIVATE
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:RelWithDebInfo>:-march=native>
  )
endif()
