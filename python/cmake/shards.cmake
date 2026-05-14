# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# Single source of truth for PyIndex shard compilation. Keep these three lists aligned by index.
set(ALAYA_PYINDEX_SHARDS
    hnsw_raw_u32
    hnsw_raw_u64
    hnsw_sq8_u32
    hnsw_sq8_u64
    hnsw_sq4_u32
    hnsw_sq4_u64
    hnsw_rabitq_u32
    hnsw_rabitq_u64
    nsg_raw_u32
    nsg_raw_u64
    nsg_sq8_u32
    nsg_sq8_u64
    nsg_sq4_u32
    nsg_sq4_u64
    nsg_rabitq_u32
    nsg_rabitq_u64
    fusion_raw_u32
    fusion_raw_u64
    fusion_sq8_u32
    fusion_sq8_u64
    fusion_sq4_u32
    fusion_sq4_u64
    fusion_rabitq_u32
    fusion_rabitq_u64
)

set(ALAYA_PYINDEX_SHARD_MACROS
    ALAYA_PYINDEX_HNSW_RAW_U32_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_RAW_U64_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_SQ8_U32_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_SQ8_U64_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_SQ4_U32_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_SQ4_U64_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_RABITQ_U32_INSTANTIATIONS
    ALAYA_PYINDEX_HNSW_RABITQ_U64_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_RAW_U32_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_RAW_U64_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_SQ8_U32_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_SQ8_U64_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_SQ4_U32_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_SQ4_U64_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_RABITQ_U32_INSTANTIATIONS
    ALAYA_PYINDEX_NSG_RABITQ_U64_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_RAW_U32_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_RAW_U64_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_SQ8_U32_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_SQ8_U64_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_SQ4_U32_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_SQ4_U64_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_RABITQ_U32_INSTANTIATIONS
    ALAYA_PYINDEX_FUSION_RABITQ_U64_INSTANTIATIONS
)

# Single strategy replicated across all shards. The list shape is retained so generate_shards.cmake's list(GET ...) loop
# and uniformity assertion work unchanged, and a future mixed strategy can be expressed by hand-writing per-shard
# entries here.
set(ALAYA_PYINDEX_SHARD_STRATEGY "core_and_search")
list(LENGTH ALAYA_PYINDEX_SHARDS _alaya_shard_count_for_strategies)
set(ALAYA_PYINDEX_SHARD_STRATEGIES "")
set(_alaya_strategy_idx 0)
while(_alaya_strategy_idx LESS _alaya_shard_count_for_strategies)
  list(APPEND ALAYA_PYINDEX_SHARD_STRATEGIES "${ALAYA_PYINDEX_SHARD_STRATEGY}")
  math(EXPR _alaya_strategy_idx "${_alaya_strategy_idx} + 1")
endwhile()
unset(_alaya_strategy_idx)
unset(_alaya_shard_count_for_strategies)
