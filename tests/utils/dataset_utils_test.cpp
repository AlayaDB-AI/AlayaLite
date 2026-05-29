// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "utils/dataset_utils.hpp"
#include "utils/hybrid_dataset_utils.hpp"
#include "utils/parser.hpp"

namespace alaya {

class DatasetTest : public ::testing::Test {
 protected:
  void SetUp() override { data_dir_ = resolve_data_dir(); }

  std::filesystem::path data_dir_;
};

TEST_F(DatasetTest, DISABLED_LoadSiftSmall) {
  auto config = sift_small(data_dir_);

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "siftsmall");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

TEST_F(DatasetTest, DISABLED_LoadDeep1M) {
  // Disabled: too slow due to large dataset download

  auto config = deep1m(data_dir_);

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "deep1M");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

TEST_F(DatasetTest, SiftMicroConfig) {
  auto config = sift_micro(data_dir_);

  EXPECT_EQ(config.name_, "siftmicro");
  EXPECT_EQ(config.max_data_num_, 1000);
  EXPECT_EQ(config.max_query_num_, 50);
  // sift_micro uses siftsmall files
  EXPECT_TRUE(config.data_file_.string().find("siftsmall") != std::string::npos);
}

TEST_F(DatasetTest, DISABLED_LoadSiftMicro) {
  auto config = sift_micro(data_dir_);
  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "siftmicro");
  // Verify data is truncated to max limits
  EXPECT_EQ(ds.data_num_, config.max_data_num_);
  EXPECT_EQ(ds.query_num_, config.max_query_num_);
  EXPECT_EQ(ds.dim_, 128);  // SIFT dimension
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);
  // Ground truth should be recomputed for truncated data
  EXPECT_EQ(ds.ground_truth_.size(), ds.query_num_ * ds.gt_dim_);
}

TEST_F(DatasetTest, DISABLED_DataTruncation) {
  // First load full siftsmall
  auto full_config = sift_small(data_dir_);
  auto full_ds = load_dataset(full_config);

  // Then load truncated version
  auto micro_config = sift_micro(data_dir_);
  auto micro_ds = load_dataset(micro_config);

  // Verify truncation
  EXPECT_LT(micro_ds.data_num_, full_ds.data_num_);
  EXPECT_LT(micro_ds.query_num_, full_ds.query_num_);

  // Verify ground truth IDs are valid (within truncated data range)
  for (uint32_t i = 0; i < micro_ds.query_num_; ++i) {
    for (uint32_t j = 0; j < micro_ds.gt_dim_; ++j) {
      uint32_t gt_id = micro_ds.ground_truth_[i * micro_ds.gt_dim_ + j];
      EXPECT_LT(gt_id, micro_ds.data_num_)
          << "GT ID " << gt_id << " exceeds data_num " << micro_ds.data_num_;
    }
  }
}

TEST(ParserTest, ParseJsonLineHandlesScalarsArraysAndEscapes) {
  auto root = parse_json_line(
      R"({"name":"alpha\nbeta\u0041","count":-42,"score":12.5,"enabled":true,"items":[1,"two"],"none":null})");
  auto *object = as_object(root);
  ASSERT_NE(object, nullptr);

  ASSERT_NE(find_any(*object, {"name"}), nullptr);
  EXPECT_EQ(*as_string(*find_any(*object, {"name"})), "alpha\nbetaA");
  EXPECT_EQ(std::get<int64_t>(find_any(*object, {"count"})->value_), -42);
  EXPECT_DOUBLE_EQ(std::get<double>(find_any(*object, {"score"})->value_), 12.5);
  EXPECT_TRUE(std::get<bool>(find_any(*object, {"enabled"})->value_));
  EXPECT_TRUE(std::holds_alternative<std::nullptr_t>(find_any(*object, {"none"})->value_));

  auto *items = as_array(*find_any(*object, {"items"}));
  ASSERT_NE(items, nullptr);
  ASSERT_EQ(items->size(), 2);
  EXPECT_FLOAT_EQ(json_to_float((*items)[0]), 1.0F);
  EXPECT_EQ(*as_string((*items)[1]), "two");
}

TEST(ParserTest, ParseJsonLineDecodesUnicodeEscapesAndSurrogatePairs) {
  auto root =
      parse_json_line(R"({"euro":"\u20AC","gclef":"\uD834\uDD1E","emoji":"\uD83D\uDE00"})");
  auto *object = as_object(root);
  ASSERT_NE(object, nullptr);

  const std::string euro = {static_cast<char>(0xE2), static_cast<char>(0x82),
                            static_cast<char>(0xAC)};
  const std::string gclef = {static_cast<char>(0xF0), static_cast<char>(0x9D),
                             static_cast<char>(0x84), static_cast<char>(0x9E)};
  const std::string emoji = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                             static_cast<char>(0x98), static_cast<char>(0x80)};

  ASSERT_NE(find_any(*object, {"euro"}), nullptr);
  ASSERT_NE(find_any(*object, {"gclef"}), nullptr);
  ASSERT_NE(find_any(*object, {"emoji"}), nullptr);
  EXPECT_EQ(*as_string(*find_any(*object, {"euro"})), euro);
  EXPECT_EQ(*as_string(*find_any(*object, {"gclef"})), gclef);
  EXPECT_EQ(*as_string(*find_any(*object, {"emoji"})), emoji);
}

TEST(ParserTest, ParseJsonLineRejectsInvalidUnicodeSurrogates) {
  EXPECT_THROW((void)parse_json_line(R"({"bad":"\uD834"})"), std::runtime_error);
  EXPECT_THROW((void)parse_json_line(R"({"bad":"\uD834\u0041"})"), std::runtime_error);
  EXPECT_THROW((void)parse_json_line(R"({"bad":"\uDD1E"})"), std::runtime_error);
}

TEST(ParserTest, LoadNpyFloatMatrixHonorsMaxRowsAndRejectsUnsupportedHeaders) {
  auto dir = std::filesystem::temp_directory_path() / "alayalite_parser_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  auto valid_file = dir / "valid.npy";
  write_tiny_npy(valid_file, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, 3, 2);
  auto matrix = load_npy_float_matrix(valid_file, 2);
  EXPECT_EQ(matrix.rows_, 2);
  EXPECT_EQ(matrix.cols_, 2);
  EXPECT_EQ(matrix.data_, (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));

  auto v2_file = dir / "valid_v2.npy";
  write_tiny_npy(v2_file, {7.0F, 8.0F}, 1, 2, "<f4", false, 2);
  auto v2_matrix = load_npy_float_matrix(v2_file);
  EXPECT_EQ(v2_matrix.rows_, 1);
  EXPECT_EQ(v2_matrix.cols_, 2);
  EXPECT_EQ(v2_matrix.data_, (std::vector<float>{7.0F, 8.0F}));

  auto invalid_descr = dir / "invalid_descr.npy";
  write_tiny_npy(invalid_descr, {1.0F, 2.0F}, 1, 2, "<f8", false);
  EXPECT_THROW(load_npy_float_matrix(invalid_descr), std::runtime_error);

  auto fortran_file = dir / "fortran.npy";
  write_tiny_npy(fortran_file, {1.0F, 2.0F}, 1, 2, "<f4", true);
  EXPECT_THROW(load_npy_float_matrix(fortran_file), std::runtime_error);

  auto invalid_json = []() {
    (void)parse_json_line(R"({"broken":[1,})");
  };
  EXPECT_THROW(invalid_json(), std::runtime_error);
  EXPECT_THROW(json_to_u32(JsonValue{int64_t{-1}}), std::runtime_error);

  std::filesystem::remove_all(dir);
}

TEST_F(DatasetTest, LoadTinyQdrantHybridDatasetSupportsAliasesAndTruncation) {
  auto dir = std::filesystem::temp_directory_path() / "alayalite_tiny_qdrant_hybrid_aliases";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  write_tiny_npy(dir / "vectors.npy",
                 {
                     1.0F,
                     0.0F,
                     0.0F,
                     1.0F,
                     0.6F,
                     0.8F,
                     -1.0F,
                     0.0F,
                 },
                 4,
                 2);

  {
    std::ofstream payloads(dir / "payloads.jsonl");
    payloads << '\n';
    payloads << R"({"tag":"keep","score":5,"group":"red","active":true})" << '\n';
    payloads << R"({"tag":"drop","score":6,"group":"blue","active":false})" << '\n';
    payloads << R"({"tag":"keep","score":8,"group":"blue","active":true})" << '\n';
    payloads << R"({"tag":"keep","score":9,"group":"green","active":true})" << '\n';
  }

  {
    std::ofstream tests(dir / "tests.jsonl");
    tests
        << R"({"query":[1.0,0.0],"filter":{"and":[{"tag":{"$eq":"keep"}},{"score":{"range":{"$gte":5,"$lt":9}}}]},"neighbors":[99,98]})"
        << '\n';
    tests
        << R"({"query_vector":[0.0,1.0],"payload_filter":{"should":[{"group":{"$in":["blue"]}},{"active":{"$eq":false}}],"must_not":[{"tag":{"$eq":"drop"}}]},"nearest_ids":[7,6]})"
        << '\n';
  }

  auto config = HybridDatasetConfig{
      .name_ = "tiny-qdrant-hybrid-aliases",
      .dir_ = dir,
      .data_file_ = dir / "vectors.npy",
      .scalar_file_ = dir / "payloads.jsonl",
      .query_file_ = dir / "tests.jsonl",
      .max_data_num_ = 3,
      .metric_ = MetricType::COS,
  };

  auto ds = load_hybrid_dataset(config);

  EXPECT_EQ(ds.data_num_, 3);
  EXPECT_EQ(ds.query_num_, 2);
  EXPECT_EQ(ds.dim_, 2);
  EXPECT_EQ(ds.gt_dim_, 2);
  EXPECT_EQ(ds.ground_truth_.size(), 4);
  EXPECT_EQ(ds.ground_truth_[0], 0U);
  EXPECT_EQ(ds.ground_truth_[1], 2U);
  EXPECT_EQ(ds.ground_truth_[2], 2U);
  EXPECT_EQ(ds.ground_truth_[3], std::numeric_limits<uint32_t>::max());

  ASSERT_EQ(ds.query_filters_.size(), 2);
  EXPECT_TRUE(ds.query_filters_[0].evaluate(ds.scalar_data_[0].metadata));
  EXPECT_FALSE(ds.query_filters_[0].evaluate(ds.scalar_data_[1].metadata));
  EXPECT_TRUE(ds.query_filters_[0].evaluate(ds.scalar_data_[2].metadata));

  EXPECT_FALSE(ds.query_filters_[1].evaluate(ds.scalar_data_[0].metadata));
  EXPECT_FALSE(ds.query_filters_[1].evaluate(ds.scalar_data_[1].metadata));
  EXPECT_TRUE(ds.query_filters_[1].evaluate(ds.scalar_data_[2].metadata));

  EXPECT_EQ(ds.indexed_fields_, (std::vector<std::string>{"active", "group", "score", "tag"}));

  std::filesystem::remove_all(dir);
}

}  // namespace alaya
