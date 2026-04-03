/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "utils/memory.hpp"

namespace alaya {

template <typename TagType = uint32_t>
class EpochVisitedSet {
 private:
  using PID = uint32_t;

  std::vector<TagType, AlignedAlloc<TagType>> tags_;
  TagType epoch_ = 1;

 public:
  static_assert(std::is_unsigned_v<TagType>, "TagType must be an unsigned integer type.");

  EpochVisitedSet() = default;
  explicit EpochVisitedSet(size_t size) : tags_(size, TagType{0}) {}

  void resize(size_t size) {
    tags_.assign(size, TagType{0});
    epoch_ = 1;
  }

  void clear() {
    ++epoch_;
    if (epoch_ == TagType{0}) {
      std::fill(tags_.begin(), tags_.end(), TagType{0});
      epoch_ = 1;
    }
  }

  [[nodiscard]] auto get(PID id) const -> bool { return tags_[id] == epoch_; }

  void set(PID id) { tags_[id] = epoch_; }

  [[nodiscard]] auto size() const -> size_t { return tags_.size(); }
};

}  // namespace alaya
