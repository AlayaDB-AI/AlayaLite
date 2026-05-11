// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <iostream>
#include <tuple>
#include <type_traits>
#include <variant>
// #include "index.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/index_type.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"

/*
namespace alaya {

using DataTypes = std::variant<float>;
using IdTypes = std::variant<uint32_t>;

template <typename... Variants>
struct MergeVariantsHelper;

template <typename... Ts1, typename... Ts2, typename... Rest>
struct MergeVariantsHelper<std::variant<Ts1...>, std::variant<Ts2...>, Rest...> {
  using type = typename MergeVariantsHelper<std::variant<Ts1..., Ts2...>, Rest...>::type;
};

template <typename... Ts>
struct MergeVariantsHelper<std::variant<Ts...>> {
  using type = std::variant<Ts...>;
};

template <typename... Variants>
using MergeVariants = typename MergeVariantsHelper<Variants...>::type;

template <typename DataType, typename IDType>
using RawSpaceVariantType =
    alaya::RawSpace<DataType, float, IDType, SequentialStorage<DataType, IDType>, 64>;

template <typename TFirst, typename TSeconds, template <typename, typename> class TClass>
struct ExpandSecond;

template <typename TFirst, typename... TSeconds, template <typename, typename> class TClass>
struct ExpandSecond<TFirst, std::variant<TSeconds...>, TClass> {
  using type = std::variant<TClass<TFirst, TSeconds>...>;
};

template <typename TFirsts, typename TSeconds, template <typename, typename> class TClass>
struct ExpandBoth;

template <typename... TFirsts, typename TSecondsTuple, template <typename, typename> class TClass>
struct ExpandBoth<std::variant<TFirsts...>, TSecondsTuple, TClass> {
  using type = MergeVariants<typename ExpandSecond<TFirsts, TSecondsTuple, TClass>::type...>;
};

template <typename Ts, template <typename> class TClass>
struct ExpandSingle;

template <typename... Ts, template <typename> class TClass>
struct ExpandSingle<std::variant<Ts...>, TClass> {
  using type = std::variant<TClass<Ts>...>;
};

template <typename Builders>
struct RawSearchIndex;

template <typename... Builders>
struct RawSearchIndex<std::variant<Builders...>> {
  using type = std::variant<
      PyIndex<Builders, RawSpace<typename Builders::DataType, typename Builders::DistanceType,
                                 typename Builders::IDType>>...>;
};

template <typename Builders>
struct SQ4SearchIndex;

template <typename... Builders>
struct SQ4SearchIndex<std::variant<Builders...>> {
  using type = std::variant<
      PyIndex<Builders, SQ4Space<typename Builders::DataType, typename Builders::DistanceType,
                                 typename Builders::IDType>>...>;
};

template <typename Builders>
struct SQ8SearchIndex;

template <typename... Builders>
struct SQ8SearchIndex<std::variant<Builders...>> {
  using type = std::variant<
      PyIndex<Builders, SQ8Space<typename Builders::DataType, typename Builders::DistanceType,
                                 typename Builders::IDType>>...>;
};

using RawSpaceVariant = typename ExpandBoth<DataTypes, IdTypes, RawSpaceVariantType>::type;
using HNSWBuilderVariant = typename ExpandSingle<RawSpaceVariant, HNSWBuilder>::type;
using NSGBuilderVariant = typename ExpandSingle<RawSpaceVariant, NSGBuilder>::type;

// using BasePyIndex =
//     MergeVariants<RawSearchIndex<HNSWBuilderVariant>::type,
//     RawSearchIndex<NSGBuilderVariant>::type,
//                   SQ4SearchIndex<HNSWBuilderVariant>::type,
//                   SQ4SearchIndex<NSGBuilderVariant>::type,
//                   SQ8SearchIndex<NSGBuilderVariant>::type,
//                   SQ8SearchIndex<HNSWBuilderVariant>::type>;

using BasePyIndex = MergeVariants<RawSearchIndex<HNSWBuilderVariant>::type>;
}  // namespace alaya
*/
