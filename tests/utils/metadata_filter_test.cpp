// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/metadata_filter_matcher.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace alaya {
namespace fs = std::filesystem;

namespace {

using TestID = uint32_t;

auto make_condition(std::string field,
                    FilterOp op,
                    MetadataValue value = int64_t(0),
                    std::vector<MetadataValue> values = {}) -> FilterCondition {
  FilterCondition condition;
  condition.field = std::move(field);
  condition.op = op;
  condition.value = std::move(value);
  condition.values = std::move(values);
  return condition;
}

auto make_sample_records() -> std::vector<ScalarData> {
  return {
      {"item_0",
       "doc_0",
       {{"category", std::string("books")},
        {"age", int64_t(10)},
        {"score", 1.5},
        {"flag", true},
        {"title", std::string("alpha guide")}}},
      {"item_1",
       "doc_1",
       {{"category", std::string("games")},
        {"age", int64_t(20)},
        {"score", 2.5},
        {"flag", false},
        {"title", std::string("beta guide")}}},
      {"item_2",
       "doc_2",
       {{"category", std::string("books")},
        {"age", int64_t(30)},
        {"score", 3.5},
        {"flag", true},
        {"title", std::string("alpha notes")}}},
      {"item_3",
       "doc_3",
       {{"category", std::string("music")},
        {"age", int64_t(40)},
        {"score", 4.5},
        {"flag", false},
        {"title", std::string("gamma notes")}}},
  };
}

void populate_storage(RocksDBStorage<TestID> &storage) {
  const auto records = make_sample_records();
  for (TestID id = 0; id < records.size(); ++id) {
    ASSERT_TRUE(storage.insert(id, records[id]));
  }
}

auto make_single_condition_filter(const std::string &field,
                                  FilterOp op,
                                  MetadataValue value = int64_t(0),
                                  std::vector<MetadataValue> values = {}) -> MetadataFilter {
  MetadataFilter filter;
  filter.conditions.push_back(make_condition(field, op, std::move(value), std::move(values)));
  return filter;
}

void expect_mask(const MetadataFilterExecutor<TestID>::BlockedBitsetResult &result,
                 const std::vector<bool> &expected) {
  ASSERT_EQ(expected.size(), result.blocked_.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(result.blocked_.get(index), expected[index]) << "Unexpected blocked bit at " << index;
  }
}

void expect_matches(const std::vector<uint8_t> &matches, const std::vector<uint8_t> &expected) {
  ASSERT_EQ(matches.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(matches[index], expected[index]) << "Unexpected match flag at " << index;
  }
}

}  // namespace

TEST(MetadataFilterConditionTest, EvaluatesAllComparisonOperators) {
  const MetadataMap metadata = {
      {"title", std::string("alpha guide")},
      {"age", int64_t(10)},
      {"score", 3.5},
      {"flag", true},
  };

  EXPECT_FALSE(make_condition("missing", FilterOp::EQ, int64_t(1)).evaluate(metadata));
  EXPECT_TRUE(make_condition("title", FilterOp::EQ, std::string("alpha guide")).evaluate(metadata));
  EXPECT_TRUE(make_condition("title", FilterOp::NE, std::string("beta guide")).evaluate(metadata));
  EXPECT_TRUE(make_condition("age", FilterOp::GT, int64_t(5)).evaluate(metadata));
  EXPECT_TRUE(make_condition("age", FilterOp::GE, int64_t(10)).evaluate(metadata));
  EXPECT_TRUE(make_condition("age", FilterOp::LT, int64_t(20)).evaluate(metadata));
  EXPECT_TRUE(make_condition("age", FilterOp::LE, int64_t(10)).evaluate(metadata));
  EXPECT_TRUE(make_condition("flag", FilterOp::GT, false).evaluate(metadata));
  EXPECT_TRUE(make_condition("title", FilterOp::GT, std::string("aardvark")).evaluate(metadata));
  EXPECT_FALSE(make_condition("age", FilterOp::GT, 9.5).evaluate(metadata));
  EXPECT_FALSE(make_condition("age", FilterOp::GE, 9.5).evaluate(metadata));
  EXPECT_FALSE(make_condition("age", FilterOp::LT, std::string("zzz")).evaluate(metadata));
  EXPECT_FALSE(make_condition("age", FilterOp::LE, std::string("zzz")).evaluate(metadata));
}

TEST(MetadataFilterConditionTest, EvaluatesCollectionAndContainsOperators) {
  const MetadataMap metadata = {
      {"category", std::string("books")},
      {"title", std::string("alpha guide")},
      {"flag", true},
  };

  EXPECT_TRUE(make_condition("category",
                             FilterOp::IN,
                             int64_t(0),
                             {std::string("books"), std::string("music")})
                  .evaluate(metadata));
  EXPECT_TRUE(make_condition("category",
                             FilterOp::NOT_IN,
                             int64_t(0),
                             {std::string("games"), std::string("video")})
                  .evaluate(metadata));
  EXPECT_TRUE(make_condition("title", FilterOp::CONTAINS, std::string("guide")).evaluate(metadata));
  EXPECT_FALSE(make_condition("flag", FilterOp::CONTAINS, std::string("true")).evaluate(metadata));
}

TEST(MetadataFilterTest, SupportsBuildersAndNestedLogic) {
  const MetadataMap metadata = {
      {"category", std::string("books")},
      {"title", std::string("alpha guide")},
      {"age", int64_t(10)},
      {"score", 3.5},
      {"flag", true},
  };

  auto empty_filter = MetadataFilter::empty();
  EXPECT_TRUE(empty_filter.is_empty());
  EXPECT_TRUE(empty_filter.evaluate(metadata));

  MetadataFilter and_filter;
  and_filter.add_eq("category", std::string("books"))
      .add_gt("score", 2.0)
      .add_ge("age", int64_t(10))
      .add_lt("age", int64_t(20))
      .add_le("age", int64_t(10))
      .add_in("title", {std::string("alpha guide"), std::string("beta guide")});
  EXPECT_TRUE(and_filter.evaluate(metadata));

  MetadataFilter or_filter;
  or_filter.logic_op = LogicOp::OR;
  or_filter.add_eq("category", std::string("games")).add_eq("title", std::string("alpha guide"));
  EXPECT_TRUE(or_filter.evaluate(metadata));

  MetadataFilter not_filter;
  not_filter.logic_op = LogicOp::NOT;
  not_filter.add_eq("category", std::string("games"));
  EXPECT_TRUE(not_filter.evaluate(metadata));

  MetadataFilter nested_filter;
  nested_filter.add_eq("category", std::string("books"));
  MetadataFilter nested_or;
  nested_or.logic_op = LogicOp::OR;
  nested_or.add_eq("title", std::string("missing")).add_eq("flag", true);
  nested_filter.add_sub_filter(std::move(nested_or));
  EXPECT_TRUE(nested_filter.evaluate(metadata));
}

TEST(MetadataFilterTest, NotWithSingleConditionNegatesCorrectly) {
  const MetadataMap metadata = {{"category", std::string("books")}};

  // NOT(category == "games") -> NOT(false) -> true
  MetadataFilter not_false;
  not_false.logic_op = LogicOp::NOT;
  not_false.add_eq("category", std::string("games"));
  EXPECT_TRUE(not_false.evaluate(metadata));

  // NOT(category == "books") -> NOT(true) -> false
  MetadataFilter not_true;
  not_true.logic_op = LogicOp::NOT;
  not_true.add_eq("category", std::string("books"));
  EXPECT_FALSE(not_true.evaluate(metadata));

  // NOT wrapping a sub-filter: NOT(score > 100) on metadata without score -> NOT(false) -> true
  MetadataFilter not_sub;
  not_sub.logic_op = LogicOp::NOT;
  MetadataFilter inner;
  inner.add_gt("score", int64_t(100));
  not_sub.add_sub_filter(std::move(inner));
  EXPECT_TRUE(not_sub.evaluate(metadata));
}

class MetadataFilterExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();

    std::replace(test_name.begin(), test_name.end(), '/', '_');
    std::replace(test_name.begin(), test_name.end(), ' ', '_');

    temp_dir_ = fs::temp_directory_path() / ("metadata_filter_executor_test_" + test_name);
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
    next_db_index_ = 0;
  }

  void TearDown() override { fs::remove_all(temp_dir_); }

  auto make_storage(std::initializer_list<std::string> indexed_fields = {})
      -> std::unique_ptr<RocksDBStorage<TestID>> {
    RocksDBConfig config;
    config.db_path_ = (temp_dir_ / ("db_" + std::to_string(next_db_index_++))).string();
    config.indexed_fields_ = std::vector<std::string>(indexed_fields);

    auto storage = std::make_unique<RocksDBStorage<TestID>>(config);
    populate_storage(*storage);
    return storage;
  }

  fs::path temp_dir_;
  size_t next_db_index_ = 0;
};

TEST_F(MetadataFilterExecutorTest, ConstructorRejectsNullStorage) {
  auto filter = MetadataFilter::empty();
  EXPECT_THROW((MetadataFilterExecutor<TestID>(filter, nullptr, 0)), std::invalid_argument);
}

TEST_F(MetadataFilterExecutorTest, EmptyFilterMatchesEverything) {
  auto storage = make_storage();
  auto filter = MetadataFilter::empty();
  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);

  EXPECT_TRUE(executor.is_trivially_true());
  EXPECT_FALSE(executor.has_index_fast_path());
  EXPECT_EQ(executor.data_num(), 4U);
  EXPECT_TRUE(executor.match(0));
  EXPECT_TRUE(executor.match(99));

  const std::vector<TestID> ids = {0, 1, 3};
  const auto subset_result = executor.build_blocked_bitset(ids);
  EXPECT_EQ(subset_result.matched_count_, ids.size());
  expect_mask(subset_result, {false, false, false});

  std::vector<uint8_t> matches;
  executor.eval_offsets(ids, matches);
  expect_matches(matches, {1, 1, 1});

  const auto full_result = executor.build_blocked_bitset();
  EXPECT_EQ(full_result.matched_count_, 4U);
  expect_mask(full_result, {false, false, false, false});
}

TEST_F(MetadataFilterExecutorTest, ExactAndInFiltersUseIndexFastPath) {
  auto storage = make_storage({"category"});

  auto exact_filter = make_single_condition_filter("category", FilterOp::EQ, std::string("books"));
  MetadataFilterExecutor<TestID> exact_executor(exact_filter, storage.get(), 4);

  EXPECT_TRUE(exact_executor.has_index_fast_path());
  EXPECT_EQ(exact_executor.indexed_ids(), (std::vector<TestID>{0, 2}));
  EXPECT_EQ(exact_executor.indexed_count(), 2U);
  EXPECT_TRUE(exact_executor.match(0));
  EXPECT_FALSE(exact_executor.match(1));

  const std::vector<TestID> ids = {0, 1, 2, 3};
  const auto result = exact_executor.build_blocked_bitset(ids);
  EXPECT_EQ(result.matched_count_, 2U);
  expect_mask(result, {false, true, false, true});

  std::vector<uint8_t> matches;
  exact_executor.eval_offsets(ids, matches);
  expect_matches(matches, {1, 0, 1, 0});

  auto in_filter = make_single_condition_filter("category",
                                                FilterOp::IN,
                                                int64_t(0),
                                                {std::string("books"),
                                                 std::string("music"),
                                                 std::string("books")});
  MetadataFilterExecutor<TestID> in_executor(in_filter, storage.get(), 4);
  EXPECT_TRUE(in_executor.has_index_fast_path());
  EXPECT_EQ(in_executor.indexed_ids(), (std::vector<TestID>{0, 2, 3}));
}

TEST_F(MetadataFilterExecutorTest, FullBitsetUsesIndexFastPathForIndexedFilters) {
  auto storage = make_storage({"category"});

  auto filter = make_single_condition_filter("category", FilterOp::EQ, std::string("books"));
  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 2U);
  expect_mask(result, {false, true, false, true});
}

TEST_F(MetadataFilterExecutorTest, IntegerRangeFiltersUseIndexFastPathAndHandleEdges) {
  auto storage = make_storage({"age"});

  auto ge_filter = make_single_condition_filter("age", FilterOp::GE, int64_t(20));
  MetadataFilterExecutor<TestID> ge_executor(ge_filter, storage.get(), 4);
  EXPECT_TRUE(ge_executor.has_index_fast_path());
  EXPECT_EQ(ge_executor.indexed_ids(), (std::vector<TestID>{1, 2, 3}));

  auto gt_filter = make_single_condition_filter("age", FilterOp::GT, int64_t(20));
  MetadataFilterExecutor<TestID> gt_executor(gt_filter, storage.get(), 4);
  EXPECT_EQ(gt_executor.indexed_ids(), (std::vector<TestID>{2, 3}));

  auto gt_max_filter = make_single_condition_filter(
      "age", FilterOp::GT, std::numeric_limits<int64_t>::max());
  MetadataFilterExecutor<TestID> gt_max_executor(gt_max_filter, storage.get(), 4);
  EXPECT_TRUE(gt_max_executor.has_index_fast_path());
  EXPECT_TRUE(gt_max_executor.indexed_ids().empty());
  EXPECT_EQ(gt_max_executor.build_blocked_bitset().matched_count_, 0U);

  auto le_filter = make_single_condition_filter("age", FilterOp::LE, int64_t(20));
  MetadataFilterExecutor<TestID> le_executor(le_filter, storage.get(), 4);
  EXPECT_EQ(le_executor.indexed_ids(), (std::vector<TestID>{0, 1}));

  auto lt_filter = make_single_condition_filter("age", FilterOp::LT, int64_t(30));
  MetadataFilterExecutor<TestID> lt_executor(lt_filter, storage.get(), 4);
  EXPECT_EQ(lt_executor.indexed_ids(), (std::vector<TestID>{0, 1}));

  auto lt_min_filter = make_single_condition_filter(
      "age", FilterOp::LT, std::numeric_limits<int64_t>::min());
  MetadataFilterExecutor<TestID> lt_min_executor(lt_min_filter, storage.get(), 4);
  EXPECT_TRUE(lt_min_executor.has_index_fast_path());
  EXPECT_TRUE(lt_min_executor.indexed_ids().empty());
}

TEST_F(MetadataFilterExecutorTest, DoubleRangeFiltersUseIndexFastPath) {
  auto storage = make_storage({"score"});

  auto ge_filter = make_single_condition_filter("score", FilterOp::GE, 2.5);
  MetadataFilterExecutor<TestID> ge_executor(ge_filter, storage.get(), 4);
  EXPECT_EQ(ge_executor.indexed_ids(), (std::vector<TestID>{1, 2, 3}));

  auto gt_filter = make_single_condition_filter("score", FilterOp::GT, 2.5);
  MetadataFilterExecutor<TestID> gt_executor(gt_filter, storage.get(), 4);
  EXPECT_EQ(gt_executor.indexed_ids(), (std::vector<TestID>{2, 3}));

  auto le_filter = make_single_condition_filter("score", FilterOp::LE, 2.5);
  MetadataFilterExecutor<TestID> le_executor(le_filter, storage.get(), 4);
  EXPECT_EQ(le_executor.indexed_ids(), (std::vector<TestID>{0, 1}));

  auto lt_filter = make_single_condition_filter("score", FilterOp::LT, 3.5);
  MetadataFilterExecutor<TestID> lt_executor(lt_filter, storage.get(), 4);
  EXPECT_EQ(lt_executor.indexed_ids(), (std::vector<TestID>{0, 1}));
}

TEST_F(MetadataFilterExecutorTest, NonIndexedAndUnsupportedFiltersFallBackToRawEvaluation) {
  auto storage = make_storage({"category"});

  auto contains_filter =
      make_single_condition_filter("title", FilterOp::CONTAINS, std::string("alpha"));
  MetadataFilterExecutor<TestID> contains_executor(contains_filter, storage.get(), 4);

  EXPECT_FALSE(contains_executor.has_index_fast_path());
  EXPECT_TRUE(contains_executor.match(0));
  EXPECT_FALSE(contains_executor.match(1));
  EXPECT_FALSE(contains_executor.match(99));

  const auto result = contains_executor.build_blocked_bitset(std::vector<TestID>{0, 1, 99, 2});
  EXPECT_EQ(result.matched_count_, 2U);
  expect_mask(result, {false, true, true, false});

  auto not_in_filter = make_single_condition_filter("category",
                                                    FilterOp::NOT_IN,
                                                    int64_t(0),
                                                    {std::string("books")});
  MetadataFilterExecutor<TestID> not_in_executor(not_in_filter, storage.get(), 4);
  EXPECT_FALSE(not_in_executor.has_index_fast_path());
  EXPECT_FALSE(not_in_executor.match(0));
  EXPECT_TRUE(not_in_executor.match(1));
}

TEST_F(MetadataFilterExecutorTest, FullBitsetFallsBackToBatchedRawEvaluationForNestedFilters) {
  auto storage = make_storage();

  MetadataFilter filter;
  filter.add_eq("category", std::string("books"));

  MetadataFilter nested_or;
  nested_or.logic_op = LogicOp::OR;
  nested_or.add_eq("title", std::string("alpha guide")).add_eq("title", std::string("alpha notes"));
  filter.add_sub_filter(std::move(nested_or));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 1026);
  EXPECT_FALSE(executor.has_index_fast_path());

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 2U);
  EXPECT_EQ(result.blocked_.size(), 1026U);
  EXPECT_FALSE(result.blocked_.get(0));
  EXPECT_TRUE(result.blocked_.get(1));
  EXPECT_FALSE(result.blocked_.get(2));
  EXPECT_TRUE(result.blocked_.get(3));
  EXPECT_TRUE(result.blocked_.get(1025));
}

TEST_F(MetadataFilterExecutorTest, IndexedRangeFiltersDisableFastPathForUnsupportedValueTypes) {
  auto storage = make_storage({"age", "score"});

  auto ge_filter = make_single_condition_filter("age", FilterOp::GE, std::string("20"));
  MetadataFilterExecutor<TestID> ge_executor(ge_filter, storage.get(), 4);
  EXPECT_FALSE(ge_executor.has_index_fast_path());

  auto gt_filter = make_single_condition_filter("age", FilterOp::GT, std::string("20"));
  MetadataFilterExecutor<TestID> gt_executor(gt_filter, storage.get(), 4);
  EXPECT_FALSE(gt_executor.has_index_fast_path());

  auto le_filter = make_single_condition_filter("score", FilterOp::LE, std::string("2.5"));
  MetadataFilterExecutor<TestID> le_executor(le_filter, storage.get(), 4);
  EXPECT_FALSE(le_executor.has_index_fast_path());

  auto lt_filter = make_single_condition_filter("score", FilterOp::LT, true);
  MetadataFilterExecutor<TestID> lt_executor(lt_filter, storage.get(), 4);
  EXPECT_FALSE(lt_executor.has_index_fast_path());
}

}  // namespace alaya
