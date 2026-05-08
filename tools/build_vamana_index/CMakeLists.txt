# ============================================================================
# tools/build_vamana_index — standalone C++ executable that depends on AlayaLite's headers.
# ============================================================================
#
# build_vamana_index — port of DiskANN's in-memory Vamana builder, producing a .index file in DiskANN's single-file
# binary layout for consumption by any downstream DiskANN-format reader.

add_executable(build_vamana_index build_vamana_for_laser.cpp)
target_link_libraries(build_vamana_index PRIVATE ${PROJECT_NAME})
target_compile_features(build_vamana_index PRIVATE cxx_std_20)

# Performance-oriented flags for the research tool. Spec task 3.5 calls for `-O3 -march=native -DNDEBUG`; the Release
# config already sets -O3/-DNDEBUG (top-level appends -Ofast which subsumes -O3), so only local native builds add
# -march=native here. CI sets ALAYA_NATIVE_ARCH=OFF so ccache never reuses host-specific objects across runner types.
if(NOT MSVC AND ALAYA_NATIVE_ARCH)
  target_compile_options(
    build_vamana_index PRIVATE $<$<CONFIG:Release>:-march=native> $<$<CONFIG:RelWithDebInfo>:-march=native>
  )
endif()
