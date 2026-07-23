// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "scalar/id_set_algebra.hpp"
#include "scalar/scalar_index_snapshot.hpp"
#include "utils/metadata_filter_matcher.hpp"

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

class FakeScalarIndex final : public ScalarIndex<TestID> {
 public:
  [[nodiscard]] auto generation() const -> uint64_t override { return 0; }

  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool override {
    return field == "category";
  }

  [[nodiscard]] auto lookup(const FilterCondition &condition) const
      -> std::optional<std::vector<TestID>> override {
    if (condition.field == "category" && condition.op == FilterOp::EQ &&
        condition.value == MetadataValue{std::string("books")}) {
      return std::vector<TestID>{0, 2};
    }
    return std::nullopt;
  }
};

class FakeRecordStore final : public RecordStore<TestID> {
 public:
  explicit FakeRecordStore(const std::vector<ScalarData> &records, uint64_t generation = 0)
      : generation_(generation) {
    for (const auto &record : records) {
      auto bytes = record.serialize();
      records_.emplace_back(bytes.begin(), bytes.end());
    }
  }

  auto get_raw_scalar(TestID id, std::string &value) const -> bool override {
    if (id >= records_.size()) {
      return false;
    }
    value = records_[id];
    return true;
  }

  [[nodiscard]] auto batch_get_raw_scalars(const std::vector<TestID> &ids) const
      -> std::vector<std::string> override {
    std::vector<std::string> result;
    result.reserve(ids.size());
    for (auto id : ids) {
      std::string value;
      get_raw_scalar(id, value);
      result.push_back(std::move(value));
    }
    return result;
  }

  [[nodiscard]] auto find_by_item_id(const std::string &item_id) const
      -> std::optional<TestID> override {
    const auto records = make_sample_records();
    for (TestID id = 0; id < records.size(); ++id) {
      if (records[id].item_id == item_id) {
        return id;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] auto size() const -> size_t override { return records_.size(); }

  [[nodiscard]] auto generation() const -> uint64_t override { return generation_; }

 private:
  std::vector<std::string> records_;  ///< Serialized canonical rows used by residual evaluation.
  uint64_t generation_ = 0;           ///< Generation exposed for provider consistency tests.
};

}  // namespace

TEST(ScalarIdSetAlgebraTest, ComputesSortedSetOperationsAndComplement) {
  EXPECT_EQ(ScalarIdSetAlgebra<TestID>::intersect({0, 2, 3}, {1, 2, 3}),
            (std::vector<TestID>{2, 3}));
  EXPECT_EQ(ScalarIdSetAlgebra<TestID>::unite({0, 2}, {1, 2, 3}),
            (std::vector<TestID>{0, 1, 2, 3}));

  auto [allow, matched_count] = ScalarIdSetAlgebra<TestID>::complement({1, 3}, 5);
  EXPECT_EQ(matched_count, 3U);
  EXPECT_TRUE(allow.get(0));
  EXPECT_FALSE(allow.get(1));
  EXPECT_TRUE(allow.get(2));
  EXPECT_FALSE(allow.get(3));
  EXPECT_TRUE(allow.get(4));
}

TEST(ScalarIndexSnapshotTest, BuildsTypedPostingsAndTracksLiveIds) {
  std::vector<ScalarIndexSnapshot<TestID>::Record> records{
      {0, ScalarData{"id_0", "doc", {{"category", "books"}, {"price", int64_t(100)}}}},
      {2, ScalarData{"id_2", "doc", {{"category", "books"}, {"price", int64_t(2500)}}}},
      {4, ScalarData{"id_4", "doc", {{"category", "games"}, {"price", int64_t(3000)}}}},
      {5, ScalarData{"id_5", "doc", {{"category", "games"}}}},
  };
  auto snapshot = ScalarIndexSnapshot<TestID>::build(7, 6, {"category", "price"}, records);

  EXPECT_EQ(snapshot->generation(), 7U);
  EXPECT_EQ(snapshot->live_count(), 4U);
  EXPECT_TRUE(snapshot->live_mask().get(0));
  EXPECT_FALSE(snapshot->live_mask().get(1));
  EXPECT_TRUE(snapshot->live_mask().get(5));
  EXPECT_EQ(snapshot->lookup(make_condition("category", FilterOp::EQ, std::string("books"))),
            (std::vector<TestID>{0, 2}));
  EXPECT_EQ(snapshot->lookup(make_condition("category", FilterOp::NE, std::string("books"))),
            (std::vector<TestID>{4, 5}));
  EXPECT_EQ(snapshot->lookup(make_condition("price", FilterOp::GT, int64_t(2000))),
            (std::vector<TestID>{2, 4}));
  EXPECT_EQ(snapshot->lookup(make_condition("price",
                                            FilterOp::NOT_IN_SET,
                                            int64_t(0),
                                            {int64_t(100), int64_t(3000)})),
            (std::vector<TestID>{2}));
  EXPECT_FALSE(
      snapshot->lookup(make_condition("document", FilterOp::EQ, std::string("doc"))).has_value());
}

TEST(ScalarIndexSnapshotTest, RejectsDuplicateAndOutOfUniverseIds) {
  auto scalar = ScalarData{"id", "doc", {{"category", "books"}}};
  EXPECT_THROW(ScalarIndexSnapshot<TestID>::build(1, 1, {"category"}, {{1, scalar}}),
               std::invalid_argument);
  EXPECT_THROW(ScalarIndexSnapshot<TestID>::build(1, 1, {"category"}, {{0, scalar}, {0, scalar}}),
               std::invalid_argument);
}

TEST(MetadataFilterExecutorProviderTest, RejectsMixedGenerations) {
  auto records = make_sample_records();
  auto snapshot = ScalarIndexSnapshot<TestID>::build(4,
                                                     records.size(),
                                                     {"category"},
                                                     {{0, records[0]},
                                                      {1, records[1]},
                                                      {2, records[2]},
                                                      {3, records[3]}});
  FakeRecordStore store(records, 3);
  auto filter = make_single_condition_filter("category", FilterOp::EQ, std::string("books"));

  EXPECT_THROW(MetadataFilterExecutor<TestID>(filter, snapshot.get(), &store, records.size()),
               std::invalid_argument);
}

TEST(MetadataFilterExecutorInterfaceTest, UsesIndexCandidatesAndRecordStoreResiduals) {
  FakeScalarIndex scalar_index;
  FakeRecordStore record_store(make_sample_records());
  MetadataFilter filter;
  filter.add_eq("category", std::string("books")).add_gt("score", 2.0);

  MetadataFilterExecutor<TestID> executor(filter,
                                          &scalar_index,
                                          &record_store,
                                          record_store.size());

  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_FALSE(executor.index_fast_path_is_exact());
  EXPECT_FALSE(executor.match(0));
  EXPECT_TRUE(executor.match(2));
  EXPECT_FALSE(executor.match(3));
  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 1U);
  expect_mask(result, {true, true, false, true});
}

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
                             FilterOp::IN_SET,
                             int64_t(0),
                             {std::string("books"), std::string("music")})
                  .evaluate(metadata));
  EXPECT_TRUE(make_condition("category",
                             FilterOp::NOT_IN_SET,
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
  EXPECT_FALSE(executor.match(99));

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
                                                FilterOp::IN_SET,
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

TEST_F(MetadataFilterExecutorTest, DirectIndexedBitsetAvoidsMaterializingCandidateIds) {
  auto storage = make_storage({"category", "age"});
  using IndexBuildMode = MetadataFilterExecutor<TestID>::IndexBuildMode;

  auto exact_filter = make_single_condition_filter("category", FilterOp::EQ, std::string("books"));
  MetadataFilterExecutor<TestID> exact_executor(exact_filter,
                                                storage.get(),
                                                4,
                                                IndexBuildMode::kSkip);
  EXPECT_FALSE(exact_executor.has_index_fast_path());
  EXPECT_TRUE(exact_executor.indexed_ids().empty());

  auto exact_result = exact_executor.build_direct_indexed_blocked_bitset();
  ASSERT_TRUE(exact_result.has_value());
  EXPECT_EQ(exact_result->matched_count_, 2U);
  expect_mask(*exact_result, {false, true, false, true});
  EXPECT_FALSE(exact_executor.has_index_fast_path());
  EXPECT_TRUE(exact_executor.indexed_ids().empty());

  auto range_filter = make_single_condition_filter("age", FilterOp::LT, int64_t(30));
  MetadataFilterExecutor<TestID> range_executor(range_filter,
                                                storage.get(),
                                                4,
                                                IndexBuildMode::kSkip);
  auto range_result = range_executor.build_direct_indexed_blocked_bitset();
  ASSERT_TRUE(range_result.has_value());
  EXPECT_EQ(range_result->matched_count_, 2U);
  expect_mask(*range_result, {false, false, true, true});
}

TEST_F(MetadataFilterExecutorTest, IndexedAndFiltersIntersectCandidateSets) {
  auto storage = make_storage({"category", "age"});

  MetadataFilter filter;
  filter.add_eq("category", std::string("books")).add_ge("age", int64_t(30));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);
  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_TRUE(executor.index_fast_path_is_exact());
  EXPECT_EQ(executor.indexed_ids(), (std::vector<TestID>{2}));

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 1U);
  expect_mask(result, {true, true, false, true});
}

TEST_F(MetadataFilterExecutorTest, IndexedOrFiltersUnionCandidateSets) {
  auto storage = make_storage({"category", "age"});

  MetadataFilter filter;
  filter.logic_op = LogicOp::OR;
  filter.add_eq("category", std::string("games")).add_ge("age", int64_t(40));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);
  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_TRUE(executor.index_fast_path_is_exact());
  EXPECT_TRUE(executor.index_fast_path_uses_materialized_ids());
  EXPECT_EQ(executor.indexed_ids(), (std::vector<TestID>{1, 3}));
  EXPECT_EQ(executor.indexed_count(), 2U);
  EXPECT_FALSE(executor.match(0));
  EXPECT_TRUE(executor.match(1));
  EXPECT_FALSE(executor.match(2));
  EXPECT_TRUE(executor.match(3));

  std::vector<TestID> visited_ids;
  executor.visit_index_fast_path_ids([&visited_ids](TestID id) {
    visited_ids.push_back(id);
  });
  EXPECT_EQ(visited_ids, (std::vector<TestID>{1, 3}));

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 2U);
  expect_mask(result, {true, false, true, false});
}

TEST_F(MetadataFilterExecutorTest, IndexedNotFiltersComplementCandidateSet) {
  auto storage = make_storage({"category"});

  MetadataFilter filter;
  filter.logic_op = LogicOp::NOT;
  filter.add_eq("category", std::string("books"));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);
  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_TRUE(executor.index_fast_path_is_exact());
  EXPECT_FALSE(executor.index_fast_path_uses_materialized_ids());
  EXPECT_TRUE(executor.indexed_ids().empty());
  EXPECT_EQ(executor.indexed_count(), 2U);
  EXPECT_FALSE(executor.match(0));
  EXPECT_TRUE(executor.match(1));
  EXPECT_FALSE(executor.match(2));
  EXPECT_TRUE(executor.match(3));

  std::vector<TestID> visited_ids;
  executor.visit_index_fast_path_ids([&visited_ids](TestID id) {
    visited_ids.push_back(id);
  });
  EXPECT_EQ(visited_ids, (std::vector<TestID>{1, 3}));

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 2U);
  expect_mask(result, {true, false, true, false});
}

TEST(MetadataFilterExecutorProviderTest, IndexedNotExcludesDeletedIdsFromComplement) {
  auto records = make_sample_records();
  auto snapshot =
      ScalarIndexSnapshot<TestID>::build(0, 3, {"category"}, {{0, records[0]}, {1, records[1]}});
  FakeRecordStore store(records);
  MetadataFilter filter;
  filter.logic_op = LogicOp::NOT;
  filter.add_eq("category", std::string("books"));

  MetadataFilterExecutor<TestID> executor(filter,
                                          snapshot.get(),
                                          &store,
                                          snapshot->universe_size(),
                                          &snapshot->live_mask(),
                                          snapshot->live_count());

  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_TRUE(executor.index_fast_path_is_exact());
  EXPECT_EQ(executor.indexed_count(), 1U);
  EXPECT_FALSE(executor.match(0));
  EXPECT_TRUE(executor.match(1));
  EXPECT_FALSE(executor.match(2));
  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 1U);
  expect_mask(result, {true, false, true});
}

TEST_F(MetadataFilterExecutorTest, IndexedAndWithResidualEvaluatesOnlyCandidateSet) {
  auto storage = make_storage({"category"});

  MetadataFilter filter;
  filter.add_eq("category", std::string("books"));
  filter.conditions.push_back(make_condition("title", FilterOp::CONTAINS, std::string("notes")));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);
  EXPECT_TRUE(executor.has_index_fast_path());
  EXPECT_FALSE(executor.index_fast_path_is_exact());
  EXPECT_EQ(executor.indexed_ids(), (std::vector<TestID>{0, 2}));
  EXPECT_FALSE(executor.match(0));
  EXPECT_TRUE(executor.match(2));

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 1U);
  expect_mask(result, {true, true, false, true});
}

TEST_F(MetadataFilterExecutorTest, OrWithUnindexedBranchFallsBackToRawEvaluation) {
  auto storage = make_storage({"category"});

  MetadataFilter filter;
  filter.logic_op = LogicOp::OR;
  filter.add_eq("category", std::string("books"));
  filter.conditions.push_back(make_condition("title", FilterOp::CONTAINS, std::string("gamma")));

  MetadataFilterExecutor<TestID> executor(filter, storage.get(), 4);
  EXPECT_FALSE(executor.has_index_fast_path());

  const auto result = executor.build_blocked_bitset();
  EXPECT_EQ(result.matched_count_, 3U);
  expect_mask(result, {false, true, false, false});
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

  auto gt_max_filter =
      make_single_condition_filter("age", FilterOp::GT, std::numeric_limits<int64_t>::max());
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

  auto lt_min_filter =
      make_single_condition_filter("age", FilterOp::LT, std::numeric_limits<int64_t>::min());
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
                                                    FilterOp::NOT_IN_SET,
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
