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
#include <pybind11/numpy.h>
#include <sys/types.h>
#include <any>
#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include "index.hpp"
#include "index/graph/fusion_graph.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/graph_concepts.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/index_type.hpp"
#include "params.hpp"
#include "reg.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/quantization_type.hpp"

namespace py = pybind11;

namespace alaya {

class Client {
 public:
  Client() = default;

  auto create_index(const std::string &name, const IndexParams &params)
      -> std::shared_ptr<PyIndexInterface> {
    auto index = std::make_shared<PyIndexInterface>(params);

    return index;
  }

  auto load_index(const std::string &name,
                  const IndexParams &params,
                  const std::string &index_path,
                  const std::string &data_path = std::string(),
                  const std::string &quant_path = std::string())
      -> std::shared_ptr<PyIndexInterface> {
    auto index = std::make_shared<PyIndexInterface>(params);
    index->load(index_path, data_path, quant_path);

    return index;
  }
};

}  // namespace alaya
