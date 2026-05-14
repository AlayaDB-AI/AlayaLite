# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

list(LENGTH ALAYA_PYINDEX_SHARDS _alaya_shard_count)
list(LENGTH ALAYA_PYINDEX_SHARD_MACROS _alaya_macro_count)
list(LENGTH ALAYA_PYINDEX_SHARD_STRATEGIES _alaya_strategy_count)

if(_alaya_shard_count EQUAL 0)
  message(FATAL_ERROR "At least one PyIndex shard must be defined")
endif()

if(NOT _alaya_shard_count EQUAL _alaya_macro_count)
  message(FATAL_ERROR "ALAYA_PYINDEX_SHARDS and ALAYA_PYINDEX_SHARD_MACROS must have equal length")
endif()

if(NOT _alaya_shard_count EQUAL _alaya_strategy_count)
  message(FATAL_ERROR "ALAYA_PYINDEX_SHARDS and ALAYA_PYINDEX_SHARD_STRATEGIES must have equal length")
endif()

math(EXPR _alaya_last_index "${_alaya_shard_count} - 1")
list(GET ALAYA_PYINDEX_SHARD_STRATEGIES 0 _alaya_uniform_strategy)
if(_alaya_last_index GREATER 0)
  set(_alaya_idx 1)
  while(_alaya_idx LESS_EQUAL _alaya_last_index)
    list(GET ALAYA_PYINDEX_SHARD_STRATEGIES ${_alaya_idx} _alaya_strategy_item)
    if(NOT _alaya_strategy_item STREQUAL _alaya_uniform_strategy)
      message(FATAL_ERROR "split_strategy must be uniform across all shards; first='${_alaya_uniform_strategy}', "
                          "index ${_alaya_idx}='${_alaya_strategy_item}'."
      )
    endif()
    math(EXPR _alaya_idx "${_alaya_idx} + 1")
  endwhile()
endif()

set(_generated_sources "")
set(_generated_instantiations_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/instantiations")
file(MAKE_DIRECTORY "${_generated_instantiations_dir}")

set(_alaya_idx 0)
while(_alaya_idx LESS_EQUAL _alaya_last_index)
  list(GET ALAYA_PYINDEX_SHARDS ${_alaya_idx} _shard_name)
  list(GET ALAYA_PYINDEX_SHARD_MACROS ${_alaya_idx} _shard_macro)
  list(GET ALAYA_PYINDEX_SHARD_STRATEGIES ${_alaya_idx} _shard_strategy)

  set(ALAYA_SHARD_NAME "${_shard_name}")
  set(ALAYA_SHARD_MACRO "${_shard_macro}")

  if(_shard_strategy STREQUAL "core_and_search")
    foreach(kind IN ITEMS core search)
      set(ALAYA_SHIM_KIND "${kind}")
      if(kind STREQUAL "core")
        set(ALAYA_SHIM_HEADER "instantiations/pyindex_instantiation_core.hpp")
      else()
        set(ALAYA_SHIM_HEADER "instantiations/pyindex_instantiation_search.hpp")
      endif()
      set(_generated_file "${_generated_instantiations_dir}/inst_${_shard_name}_${kind}.cpp")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/inst_shim.cpp.in" "${_generated_file}" @ONLY)
      list(APPEND _generated_sources "${_generated_file}")
    endforeach()
  elseif(_shard_strategy STREQUAL "single_tu")
    set(ALAYA_SHIM_KIND "single")
    set(ALAYA_SHIM_HEADER "instantiations/pyindex_instantiation_single.hpp")
    set(_generated_file "${_generated_instantiations_dir}/inst_${_shard_name}.cpp")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/inst_shim.cpp.in" "${_generated_file}" @ONLY)
    list(APPEND _generated_sources "${_generated_file}")
  else()
    message(
      FATAL_ERROR
        "Unknown split strategy '${_shard_strategy}' for shard '${_shard_name}'. Expected core_and_search or single_tu."
    )
  endif()
  math(EXPR _alaya_idx "${_alaya_idx} + 1")
endwhile()
