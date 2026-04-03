/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

// todo: for raw_space and rabitq_space, reuse the same build/search space to avoid duplicate data
// storage and fitting work in materialized-view partitions.
// todo: for sq4/sq8 space, model the raw build space and quantized search space more explicitly.

// when scalar storage is unnecessary, strip scalar payloads all the way through search space.
template <typename SpaceType>
struct StripScalarData;

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = RawSpace<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<SQ4Space<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = SQ4Space<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<SQ8Space<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = SQ8Space<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType, typename DistanceType, typename IDType, typename ScalarDataType>
struct StripScalarData<RaBitQSpace<DataType, DistanceType, IDType, ScalarDataType>> {
  using type = RaBitQSpace<DataType, DistanceType, IDType, EmptyScalarData>;
};

template <typename SpaceType>
using StripScalarDataT = typename StripScalarData<SpaceType>::type;

struct MaterializedViewPartitionSelection {
  bool eligible_ = false;
  bool filter_covered_ = false;
  std::vector<MetadataValue> values_;
};

inline void append_unique_metadata_value(std::vector<MetadataValue> &values,
                                         const MetadataValue &value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

inline auto intersect_metadata_values(const std::vector<MetadataValue> &lhs,
                                      const std::vector<MetadataValue> &rhs)
    -> std::vector<MetadataValue> {
  std::vector<MetadataValue> intersection;
  for (const auto &value : lhs) {
    if (std::find(rhs.begin(), rhs.end(), value) != rhs.end()) {
      append_unique_metadata_value(intersection, value);
    }
  }
  return intersection;
}

inline auto collect_conjunctive_filter_conditions(const MetadataFilter &filter,
                                                  std::vector<const FilterCondition *> &conditions)
    -> bool {
  if (filter.is_empty()) {
    return true;
  }
  if (filter.logic_op != LogicOp::AND) {
    return false;
  }

  for (const auto &condition : filter.conditions) {
    conditions.push_back(&condition);
  }
  for (const auto &sub_filter : filter.sub_filters) {
    if (sub_filter == nullptr || !collect_conjunctive_filter_conditions(*sub_filter, conditions)) {
      return false;
    }
  }
  return true;
}

inline auto analyze_materialized_view_filter(const MetadataFilter &filter,
                                             const std::string &target_field)
    -> MaterializedViewPartitionSelection {
  if (filter.is_empty() || target_field.empty()) {
    return {};
  }

  std::vector<const FilterCondition *> conditions;
  if (!collect_conjunctive_filter_conditions(filter, conditions)) {
    return {};
  }

  std::optional<std::vector<MetadataValue>> allowed_values;
  bool filter_covered = true;
  for (const auto *condition : conditions) {
    if (condition->field != target_field) {
      filter_covered = false;
      continue;
    }

    std::vector<MetadataValue> condition_values;
    switch (condition->op) {
      case FilterOp::EQ:
        append_unique_metadata_value(condition_values, condition->value);
        break;
      case FilterOp::IN:
        for (const auto &value : condition->values) {
          append_unique_metadata_value(condition_values, value);
        }
        break;
      default:
        return {};
    }

    if (!allowed_values.has_value()) {
      allowed_values = std::move(condition_values);
    } else {
      allowed_values = intersect_metadata_values(*allowed_values, condition_values);
    }
  }

  if (!allowed_values.has_value()) {
    return {};
  }

  return MaterializedViewPartitionSelection{true, filter_covered, std::move(*allowed_values)};
}

}  // namespace alaya
