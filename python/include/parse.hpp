// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "executor/search_info.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

namespace py = pybind11;

template <typename T>
auto get_topk_array(const std::vector<std::vector<T>> &res_pool, size_t topk) -> py::array_t<T> {
  size_t query_size = res_pool.size();
  if (query_size == 0 || topk == 0) {
    return py::array_t<T>({query_size, topk});
  }

  py::array_t<T> ret({query_size, topk});
  T *ret_data = ret.mutable_data();

  size_t output_row_byte_stride = topk * sizeof(T);
  for (size_t i = 0; i < query_size; ++i) {
    std::memcpy(ret_data + (i * topk), res_pool[i].data(), output_row_byte_stride);
  }
  return ret;
}

inline auto pydict_to_metadata_map(const py::dict &meta) -> MetadataMap {
  MetadataMap meta_map;
  for (auto item : meta) {
    std::string key = py::str(item.first);
    auto value = item.second;
    if (py::isinstance<py::bool_>(value)) {
      meta_map[key] = value.cast<bool>();
    } else if (py::isinstance<py::int_>(value)) {
      meta_map[key] = value.cast<int64_t>();
    } else if (py::isinstance<py::float_>(value)) {
      meta_map[key] = value.cast<double>();
    } else if (py::isinstance<py::str>(value)) {
      meta_map[key] = value.cast<std::string>();
    }
  }
  return meta_map;
}

inline auto metadata_map_to_pydict(const MetadataMap &meta_map) -> py::dict {
  py::dict meta;

  for (const auto &item : meta_map) {
    // Use explicit variable declaration instead of structured binding for wider compiler support.
    const auto &key = item.first;
    const auto &value = item.second;
    std::visit(
        [&meta, &key](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, bool>) {
            meta[key.c_str()] = py::bool_(arg);
          } else if constexpr (std::is_same_v<T, int64_t>) {
            meta[key.c_str()] = py::int_(arg);
          } else if constexpr (std::is_same_v<T, double>) {
            meta[key.c_str()] = py::float_(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            meta[key.c_str()] = py::str(arg);
          }
        },
        value);
  }
  return meta;
}

inline auto parse_filter_exec_hint(std::string_view hint) -> FilterExecHint {
  if (hint.empty() || hint == "auto") {
    return FilterExecHint::kAuto;
  }
  if (hint == "disable" || hint == "bitset_prefilter") {
    return FilterExecHint::kDisableIterative;
  }
  if (hint == "iterative_filter") {
    return FilterExecHint::kIterativeFilter;
  }
  throw std::runtime_error(
      "filter_execution_hint must be one of: '', 'auto', 'disable', 'bitset_prefilter', "
      "'iterative_filter'");
}

inline auto scalar_data_to_pydict(const ScalarData &scalar_data) -> py::dict {
  py::dict result;
  result["item_id"] = scalar_data.item_id;
  result["document"] = scalar_data.document;
  result["metadata"] = metadata_map_to_pydict(scalar_data.metadata);
  return result;
}

inline auto build_scalar_data_vec(const py::list &item_ids,
                                  const py::object &documents,
                                  const py::object &metadata_list,
                                  size_t count) -> std::vector<ScalarData> {
  std::vector<ScalarData> scalar_data_vec;
  scalar_data_vec.reserve(count);

  py::list docs = documents.is_none() ? py::list() : documents.cast<py::list>();
  py::list metas = metadata_list.is_none() ? py::list() : metadata_list.cast<py::list>();
  std::unordered_set<std::string> seen_item_ids;

  for (size_t i = 0; i < count; i++) {
    MetadataMap meta_map;
    if (i < metas.size()) {
      meta_map = pydict_to_metadata_map(metas[i].cast<py::dict>());
    }
    std::string doc = (i < docs.size()) ? docs[i].cast<std::string>() : "";
    auto item_id_str = py::str(item_ids[i]).cast<std::string>();
    if (!item_id_str.empty() && !seen_item_ids.insert(item_id_str).second) {
      throw std::runtime_error("Duplicate item_id: " + item_id_str);
    }
    scalar_data_vec.emplace_back(item_id_str, doc, std::move(meta_map));
  }
  return scalar_data_vec;
}

}  // namespace alaya
