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

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "index/disk/types.hpp"

namespace alaya::disk {

TEST(DiskTypesRoundTrip, EnumStringRoundTrip) {
  EXPECT_EQ(index_type_to_string(DiskIndexType::Flat), "disk_flat");
  EXPECT_EQ(index_type_to_string(DiskIndexType::Vamana), "disk_vamana");
  EXPECT_EQ(index_type_to_string(DiskIndexType::Laser), "disk_laser");

  EXPECT_EQ(index_type_from_string("disk_flat"), DiskIndexType::Flat);
  EXPECT_EQ(index_type_from_string("disk_vamana"), DiskIndexType::Vamana);
  EXPECT_EQ(index_type_from_string("disk_laser"), DiskIndexType::Laser);
}

TEST(DiskTypesUnknownIndexTypeStringRejected, ThrowsWithOffendingValueInMessage) {
  try {
    (void)index_type_from_string("disk_bogus");
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_bogus"), std::string::npos)
        << "exception message must contain the offending value 'disk_bogus', got: " << msg;
  }
}

TEST(DiskSearchOptionsDefaults, MatchSpecValues) {
  const DiskSearchOptions opts{};
  EXPECT_EQ(opts.top_k, 10u);
  EXPECT_EQ(opts.ef, 100u);
  EXPECT_EQ(opts.beam_width, 4u);
  EXPECT_TRUE(opts.exact_rerank);
}

namespace {

class StubSearcher : public SegmentSearcher {
 public:
  ~StubSearcher() override = default;

  auto search(const float * /*query*/, const DiskSearchOptions & /*opts*/) const
      -> std::vector<DiskSearchHit> override {
    return {};
  }
  auto size() const -> uint64_t override { return 0; }
  auto dim() const -> uint32_t override { return 0; }
  auto type() const -> DiskIndexType override { return DiskIndexType::Flat; }
};

}  // namespace

TEST(SegmentSearcherInterfaceShape, AbstractWithFourPureVirtualsPlusVDtor) {
  static_assert(std::is_abstract_v<SegmentSearcher>,
                "SegmentSearcher must be abstract via pure virtuals");
  static_assert(std::has_virtual_destructor_v<SegmentSearcher>,
                "SegmentSearcher must have a virtual destructor");
  // Abstract classes are reported as non-copy/move-constructible by the type
  // traits (you cannot construct an instance of an abstract type), which is
  // the property the spec actually cares about.
  static_assert(!std::is_copy_constructible_v<SegmentSearcher>);
  static_assert(!std::is_move_constructible_v<SegmentSearcher>);

  StubSearcher s;
  EXPECT_EQ(s.type(), DiskIndexType::Flat);
  EXPECT_EQ(s.size(), 0u);
  EXPECT_EQ(s.dim(), 0u);

  const SegmentSearcher &base = s;
  EXPECT_EQ(base.type(), DiskIndexType::Flat);
}

}  // namespace alaya::disk
