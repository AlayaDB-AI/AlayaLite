// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "storage/rocksdb_record_store.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {
// NOLINTBEGIN
namespace fs = std::filesystem;

namespace {

template <typename T>
void append_raw(std::vector<char> &buffer, const T &value) {
  const char *ptr = reinterpret_cast<const char *>(&value);
  buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
}

void append_string(std::vector<char> &buffer, std::string_view value) {
  auto len = static_cast<uint32_t>(value.size());
  append_raw(buffer, len);
  buffer.insert(buffer.end(), value.begin(), value.end());
}

auto make_scalar_payload_prefix() -> std::vector<char> {
  std::vector<char> payload;
  append_string(payload, "id_001");
  append_string(payload, "document content");
  return payload;
}

auto make_scalar_payload_with_metadata_header(uint32_t count) -> std::vector<char> {
  auto payload = make_scalar_payload_prefix();
  append_raw(payload, count);
  return payload;
}

auto vector_bytes(float value) -> std::string {
  return {reinterpret_cast<const char *>(&value), sizeof(value)};
}

auto make_condition(std::string field,
                    FilterOp op,
                    MetadataValue value,
                    std::vector<MetadataValue> values = {}) -> FilterCondition {
  return FilterCondition{std::move(field), op, std::move(value), std::move(values)};
}

/** @brief Overwrite one v3 metadata value to exercise restart corruption checks. */
void overwrite_record_store_metadata(const fs::path &path,
                                     std::string_view key,
                                     std::string_view value) {
  rocksdb::Options options;
  options.create_if_missing = false;
  auto names = RocksDBRecordStoreSchema::column_families();
  std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
  descriptors.reserve(names.size());
  for (const auto &name : names) {
    descriptors.emplace_back(name, rocksdb::ColumnFamilyOptions(options));
  }

  std::vector<rocksdb::ColumnFamilyHandle *> handles;
  rocksdb::DB *db = nullptr;
  auto status =
      rocksdb::DB::Open(rocksdb::DBOptions(options), path.string(), descriptors, &handles, &db);
  ASSERT_TRUE(status.ok()) << status.ToString();
  ASSERT_EQ(handles.size(), names.size());

  auto metadata = std::find(names.begin(), names.end(), RocksDBRecordStoreSchema::kMetadata);
  ASSERT_NE(metadata, names.end());
  auto metadata_index = static_cast<size_t>(std::distance(names.begin(), metadata));
  status = db->Put(rocksdb::WriteOptions{}, handles[metadata_index], key, value);
  EXPECT_TRUE(status.ok()) << status.ToString();

  for (auto *handle : handles) {
    EXPECT_TRUE(db->DestroyColumnFamilyHandle(handle).ok());
  }
  EXPECT_TRUE(db->Close().ok());
  delete db;
}

}  // namespace

class RocksDBStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();

    std::replace(test_name.begin(), test_name.end(), '/', '_');
    std::replace(test_name.begin(), test_name.end(), ' ', '_');

    temp_dir_ = fs::temp_directory_path() / ("rocksdb_test_" + test_name);
    fs::create_directories(temp_dir_);
    config_.db_path_ = temp_dir_.string();
  }

  void TearDown() override {
    if (fs::exists(temp_dir_)) {
      fs::remove_all(temp_dir_);
    }
  }

  RocksDBConfig config_;
  fs::path temp_dir_;
};

class RocksDBRecordStoreTest : public RocksDBStorageTest {
 protected:
  auto record_store_config(const fs::path &path = {}) const -> RocksDBRecordStoreConfig {
    RocksDBRecordStoreConfig config;
    config.db_path_ = (path.empty() ? temp_dir_ / "record_store" : path).string();
    config.indexed_fields_ = {"category", "price"};
    return config;
  }
};

TEST_F(RocksDBRecordStoreTest, PersistsCompleteRowsAndRebuildsOneGenerationOnRestart) {
  auto path = temp_dir_ / "record_store";
  {
    RocksDBRecordStore<> store(record_store_config(path));
    ScalarData scalar{"item-7",
                      "document",
                      {{"category", std::string("target")}, {"price", int64_t(2500)}}};
    auto raw = vector_bytes(7.0F);
    ASSERT_TRUE(store.upsert(7, scalar, raw));
    EXPECT_EQ(store.generation(), 1U);
    EXPECT_EQ(store.size(), 1U);
    EXPECT_EQ(store.universe_size(), 8U);
    EXPECT_EQ(store.find_by_item_id("item-7"), 7U);

    std::string stored_raw;
    EXPECT_TRUE(store.get_raw_vector(7, stored_raw));
    EXPECT_EQ(stored_raw, raw);

    auto view = store.acquire_query_view();
    EXPECT_EQ(view->generation(), store.generation());
    EXPECT_EQ(view->scalar_index().lookup(
                  make_condition("category", FilterOp::EQ, std::string("target"))),
              (std::vector<uint32_t>{7}));
    EXPECT_EQ(view->scalar_index().lookup(make_condition("price", FilterOp::GT, int64_t(2000))),
              (std::vector<uint32_t>{7}));
  }

  RocksDBRecordStore<> reopened(record_store_config(path));
  EXPECT_EQ(reopened.generation(), 1U);
  EXPECT_EQ(reopened.size(), 1U);
  EXPECT_EQ(reopened.universe_size(), 8U);
  EXPECT_EQ(reopened.current_scalar_snapshot()->live_count(), 1U);
  EXPECT_TRUE(reopened.current_scalar_snapshot()->live_mask().get(7));
}

TEST_F(RocksDBRecordStoreTest, RejectsIndexedFieldConfigurationChangesOnRestart) {
  auto path = temp_dir_ / "record_store";
  { RocksDBRecordStore<> store(record_store_config(path)); }

  auto changed = record_store_config(path);
  changed.indexed_fields_ = {"different"};
  EXPECT_THROW(RocksDBRecordStore<>{changed}, std::runtime_error);
}

TEST_F(RocksDBRecordStoreTest, RejectsPriorSchemaInsteadOfMigratingIt) {
  auto path = temp_dir_ / "record_store";
  { RocksDBRecordStore<> store(record_store_config(path)); }

  overwrite_record_store_metadata(path, "schema_version", "2");
  EXPECT_THROW(RocksDBRecordStore<>(record_store_config(path)), std::runtime_error);
}

TEST_F(RocksDBRecordStoreTest, RejectsPersistedUniverseSmallerThanCanonicalRecordIds) {
  auto path = temp_dir_ / "record_store";
  {
    RocksDBRecordStore<> store(record_store_config(path));
    ASSERT_TRUE(store.upsert(7, ScalarData{"item-7", "document", {}}, vector_bytes(7.0F)));
  }

  overwrite_record_store_metadata(path, "universe_size", "7");
  EXPECT_THROW(RocksDBRecordStore<>(record_store_config(path)), std::invalid_argument);
}

TEST_F(RocksDBRecordStoreTest, QueryViewKeepsScalarAndVectorReadsAtItsCapturedGeneration) {
  RocksDBRecordStore<> store(record_store_config());
  ASSERT_TRUE(store.upsert(0,
                           ScalarData{"item-0", "old-document", {{"category", std::string("old")}}},
                           vector_bytes(1.0F)));
  auto old_view = store.acquire_query_view();

  ASSERT_TRUE(store.upsert(0,
                           ScalarData{"item-0", "new-document", {{"category", std::string("new")}}},
                           vector_bytes(2.0F)));
  auto new_view = store.acquire_query_view();

  EXPECT_EQ(old_view->generation(), 1U);
  EXPECT_EQ(new_view->generation(), 2U);
  std::string old_scalar_raw;
  std::string new_scalar_raw;
  ASSERT_TRUE(old_view->get_raw_scalar(0, old_scalar_raw));
  ASSERT_TRUE(new_view->get_raw_scalar(0, new_scalar_raw));
  EXPECT_EQ(ScalarData::deserialize(old_scalar_raw.data(), old_scalar_raw.size()).document,
            "old-document");
  EXPECT_EQ(ScalarData::deserialize(new_scalar_raw.data(), new_scalar_raw.size()).document,
            "new-document");
  EXPECT_EQ(old_view->scalar_index().lookup(
                make_condition("category", FilterOp::EQ, std::string("old"))),
            (std::vector<uint32_t>{0}));
  EXPECT_EQ(new_view->scalar_index().lookup(
                make_condition("category", FilterOp::EQ, std::string("new"))),
            (std::vector<uint32_t>{0}));

  std::string old_vector;
  std::string new_vector;
  EXPECT_TRUE(old_view->get_raw_vector(0, old_vector));
  EXPECT_TRUE(new_view->get_raw_vector(0, new_vector));
  EXPECT_EQ(old_vector, vector_bytes(1.0F));
  EXPECT_EQ(new_vector, vector_bytes(2.0F));
}

TEST_F(RocksDBRecordStoreTest, RejectsDuplicateItemIdsAndDeletesEveryRowComponent) {
  RocksDBRecordStore<> store(record_store_config());
  ASSERT_TRUE(store.upsert(0,
                           ScalarData{"same", "doc-0", {{"category", std::string("a")}}},
                           vector_bytes(0.0F)));
  EXPECT_FALSE(store.upsert(1,
                            ScalarData{"same", "doc-1", {{"category", std::string("b")}}},
                            vector_bytes(1.0F)));
  EXPECT_EQ(store.generation(), 1U);
  EXPECT_EQ(store.size(), 1U);

  ASSERT_TRUE(store.remove(0));
  EXPECT_EQ(store.generation(), 2U);
  EXPECT_EQ(store.size(), 0U);
  EXPECT_EQ(store.universe_size(), 1U);
  EXPECT_FALSE(store.find_by_item_id("same").has_value());
  std::string value;
  EXPECT_FALSE(store.get_raw_scalar(0, value));
  EXPECT_FALSE(store.get_raw_vector(0, value));
  EXPECT_FALSE(store.current_scalar_snapshot()->live_mask().get(0));
}

TEST_F(RocksDBRecordStoreTest, CheckpointRestoresAllColumnFamiliesAtOneGeneration) {
  auto checkpoint = temp_dir_ / "checkpoint";
  {
    RocksDBRecordStore<> store(record_store_config());
    ASSERT_TRUE(store.upsert(3,
                             ScalarData{"item-3", "doc", {{"category", std::string("target")}}},
                             vector_bytes(3.0F)));
    store.save_checkpoint(checkpoint.string());
  }

  auto restored_config = record_store_config(checkpoint);
  restored_config.create_if_missing_ = false;
  RocksDBRecordStore<> restored(restored_config);
  EXPECT_EQ(restored.generation(), 1U);
  EXPECT_EQ(restored.size(), 1U);
  EXPECT_EQ(restored.find_by_item_id("item-3"), 3U);
  std::string raw;
  EXPECT_TRUE(restored.get_raw_vector(3, raw));
  EXPECT_EQ(raw, vector_bytes(3.0F));
}

TEST_F(RocksDBRecordStoreTest, RefusesLegacyDirectoryWithoutModifyingIt) {
  auto legacy_path = temp_dir_ / "legacy";
  auto legacy_config = config_;
  legacy_config.db_path_ = legacy_path.string();
  {
    RocksDBStorage<> legacy(legacy_config);
    ASSERT_TRUE(legacy.insert(0, ScalarData{"legacy-item", "doc", {}}));
  }

  EXPECT_THROW(RocksDBRecordStore<>(record_store_config(legacy_path)), std::runtime_error);
  RocksDBStorage<> reopened_legacy(legacy_config);
  EXPECT_EQ(reopened_legacy.count(), 1U);
  EXPECT_EQ(reopened_legacy[0].item_id, "legacy-item");
}

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BasicInsertAndGet) {
  RocksDBStorage<> storage(config_);

  ScalarData data{"id_001", "document content", {{"category", "tech"}, {"score", int64_t(95)}}};
  bool success = storage.insert(0, data);
  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 1U);

  auto retrieved = storage[0];
  EXPECT_EQ(retrieved.item_id, "id_001");
  EXPECT_EQ(retrieved.document, "document content");
  EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("category")), "tech");
  EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("score")), 95);
}

TEST_F(RocksDBStorageTest, InvalidIDReturnsEmpty) {
  RocksDBStorage<> storage(config_);
  auto invalid_data = storage[999];
  EXPECT_TRUE(invalid_data.item_id.empty());
  EXPECT_TRUE(invalid_data.document.empty());
  EXPECT_TRUE(invalid_data.metadata.empty());
}

TEST_F(RocksDBStorageTest, IsValidWorksCorrectly) {
  RocksDBStorage<> storage(config_);
  ScalarData data{"test_id", "test doc", {}};
  storage.insert(0, data);
  EXPECT_TRUE(storage.is_valid(0));
  EXPECT_FALSE(storage.is_valid(1));
  EXPECT_FALSE(storage.is_valid(999));
}

TEST(ScalarDataTest, DeserializeSingleMetadataValueAndSelectedFields) {
  ScalarData data{"id_001",
                  "document content",
                  {{"category", std::string("tech")}, {"score", int64_t(95)}, {"featured", true}}};
  auto serialized = data.serialize();

  std::unordered_set<std::string> required_fields{"category"};
  auto selected = ScalarData::deserialize_selected_metadata(serialized.data(),
                                                            serialized.size(),
                                                            required_fields);
  ASSERT_EQ(selected.size(), 1U);
  EXPECT_EQ(std::get<std::string>(selected.at("category")), "tech");

  auto score =
      ScalarData::deserialize_single_metadata_value(serialized.data(), serialized.size(), "score");
  ASSERT_TRUE(score.has_value());
  EXPECT_EQ(std::get<int64_t>(*score), 95);

  auto missing = ScalarData::deserialize_single_metadata_value(serialized.data(),
                                                               serialized.size(),
                                                               "missing");
  EXPECT_FALSE(missing.has_value());
}

TEST(ScalarDataTest, SelectiveMetadataDeserializationHandlesAllValueTypes) {
  auto payload = make_scalar_payload_with_metadata_header(4);

  append_string(payload, "int_value");
  payload.push_back(0);
  append_raw(payload, int64_t(95));

  append_string(payload, "double_value");
  payload.push_back(1);
  append_raw(payload, 3.5);

  append_string(payload, "string_value");
  payload.push_back(2);
  append_string(payload, "tech");

  append_string(payload, "bool_value");
  payload.push_back(3);
  append_raw(payload, true);

  auto selected =
      ScalarData::deserialize_selected_metadata(payload.data(),
                                                payload.size(),
                                                std::unordered_set<std::string>{"string_value",
                                                                                "bool_value"});
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_EQ(std::get<std::string>(selected.at("string_value")), "tech");
  EXPECT_TRUE(std::get<bool>(selected.at("bool_value")));

  auto int_value =
      ScalarData::deserialize_single_metadata_value(payload.data(), payload.size(), "int_value");
  ASSERT_TRUE(int_value.has_value());
  EXPECT_EQ(std::get<int64_t>(*int_value), 95);

  auto double_value =
      ScalarData::deserialize_single_metadata_value(payload.data(), payload.size(), "double_value");
  ASSERT_TRUE(double_value.has_value());
  EXPECT_DOUBLE_EQ(std::get<double>(*double_value), 3.5);

  auto string_value =
      ScalarData::deserialize_single_metadata_value(payload.data(), payload.size(), "string_value");
  ASSERT_TRUE(string_value.has_value());
  EXPECT_EQ(std::get<std::string>(*string_value), "tech");

  auto bool_value =
      ScalarData::deserialize_single_metadata_value(payload.data(), payload.size(), "bool_value");
  ASSERT_TRUE(bool_value.has_value());
  EXPECT_TRUE(std::get<bool>(*bool_value));
}

TEST(ScalarDataTest, SelectiveMetadataDeserializationRejectsInvalidInputs) {
  auto payload = make_scalar_payload_with_metadata_header(1);
  append_string(payload, "score");
  payload.push_back(0);
  append_raw(payload, int64_t(95));

  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(payload.data(),
                                                        payload.size(),
                                                        std::unordered_set<std::string>{})
                  .empty());
  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(nullptr,
                                                        payload.size(),
                                                        std::unordered_set<std::string>{"score"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(payload.data(), payload.size(), "")
                   .has_value());
  EXPECT_FALSE(
      ScalarData::deserialize_single_metadata_value(nullptr, payload.size(), "score").has_value());

  auto missing_metadata_count = make_scalar_payload_prefix();
  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(missing_metadata_count.data(),
                                                        missing_metadata_count.size(),
                                                        std::unordered_set<std::string>{"score"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(missing_metadata_count.data(),
                                                             missing_metadata_count.size(),
                                                             "score")
                   .has_value());
}

TEST(ScalarDataTest, SelectiveMetadataDeserializationReturnsEmptyOnCorruptPayloads) {
  auto truncated_key = make_scalar_payload_with_metadata_header(1);
  append_raw(truncated_key, uint32_t(32));
  truncated_key.insert(truncated_key.end(), {'s', 'c', 'o', 'r', 'e'});

  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(truncated_key.data(),
                                                        truncated_key.size(),
                                                        std::unordered_set<std::string>{"score"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(truncated_key.data(),
                                                             truncated_key.size(),
                                                             "score")
                   .has_value());

  auto truncated_required_value = make_scalar_payload_with_metadata_header(1);
  append_string(truncated_required_value, "score");
  truncated_required_value.push_back(0);
  truncated_required_value.push_back('\x01');
  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(truncated_required_value.data(),
                                                        truncated_required_value.size(),
                                                        std::unordered_set<std::string>{"score"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(truncated_required_value.data(),
                                                             truncated_required_value.size(),
                                                             "score")
                   .has_value());

  auto truncated_skipped_value = make_scalar_payload_with_metadata_header(1);
  append_string(truncated_skipped_value, "score");
  truncated_skipped_value.push_back(0);
  truncated_skipped_value.push_back('\x01');
  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(truncated_skipped_value.data(),
                                                        truncated_skipped_value.size(),
                                                        std::unordered_set<std::string>{"other"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(truncated_skipped_value.data(),
                                                             truncated_skipped_value.size(),
                                                             "other")
                   .has_value());

  auto unknown_type = make_scalar_payload_with_metadata_header(1);
  append_string(unknown_type, "score");
  unknown_type.push_back(99);
  EXPECT_TRUE(ScalarData::deserialize_selected_metadata(unknown_type.data(),
                                                        unknown_type.size(),
                                                        std::unordered_set<std::string>{"other"})
                  .empty());
  EXPECT_FALSE(ScalarData::deserialize_single_metadata_value(unknown_type.data(),
                                                             unknown_type.size(),
                                                             "other")
                   .has_value());
}

TEST(ScalarDataTest, DeserializeRejectsTruncatedPayload) {
  ScalarData data{"id_001", "document content", {{"category", std::string("tech")}}};
  auto serialized = data.serialize();
  serialized.pop_back();

  EXPECT_THROW(ScalarData::deserialize(nullptr, 0), std::runtime_error);
  EXPECT_THROW(ScalarData::deserialize(serialized.data(), serialized.size()), std::runtime_error);
}

TEST_F(RocksDBStorageTest, UpdateOperations) {
  RocksDBStorage<> storage(config_);

  ScalarData data1{"id_001", "original", {{"version", int64_t(1)}}};
  storage.insert(0, data1);
  EXPECT_EQ(storage[0].document, "original");

  ScalarData data2{"id_001", "updated", {{"version", int64_t(2)}}};
  bool success = storage.update(0, data2);
  EXPECT_TRUE(success);

  auto retrieved = storage[0];
  EXPECT_EQ(retrieved.document, "updated");
  EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("version")), 2);
}

TEST_F(RocksDBStorageTest, InsertRejectsDuplicateItemIdAndKeepsOriginalRecord) {
  RocksDBStorage<> storage(config_);

  ASSERT_TRUE(storage.insert(0, ScalarData{"dup", "original", {{"version", int64_t(1)}}}));

  EXPECT_FALSE(storage.insert(1, ScalarData{"dup", "duplicate", {{"version", int64_t(2)}}}));

  EXPECT_EQ(storage.count(), 1U);
  EXPECT_TRUE(storage.is_valid(0));
  EXPECT_FALSE(storage.is_valid(1));
  EXPECT_EQ(storage[0].document, "original");

  auto owner = storage.find_by_item_id("dup");
  ASSERT_TRUE(owner.has_value());
  EXPECT_EQ(owner.value(), 0U);
}

TEST_F(RocksDBStorageTest, EmptyItemIdsAreNotUniqueKeys) {
  RocksDBStorage<> storage(config_);

  EXPECT_TRUE(storage.insert(0, ScalarData{"", "first", {}}));
  EXPECT_TRUE(storage.insert(1, ScalarData{"", "second", {}}));

  EXPECT_EQ(storage.count(), 2U);
  EXPECT_TRUE(storage[0].item_id.empty());
  EXPECT_TRUE(storage[1].item_id.empty());
}

TEST_F(RocksDBStorageTest, UpdateRejectsItemIdOwnedByAnotherRecord) {
  RocksDBStorage<> storage(config_);

  ASSERT_TRUE(storage.insert(0, ScalarData{"item_a", "doc a", {}}));
  ASSERT_TRUE(storage.insert(1, ScalarData{"item_b", "doc b", {}}));

  EXPECT_FALSE(storage.update(1, ScalarData{"item_a", "doc b updated", {}}));

  EXPECT_EQ(storage.count(), 2U);
  EXPECT_EQ(storage[1].item_id, "item_b");
  EXPECT_EQ(storage[1].document, "doc b");

  auto item_a_owner = storage.find_by_item_id("item_a");
  auto item_b_owner = storage.find_by_item_id("item_b");
  ASSERT_TRUE(item_a_owner.has_value());
  ASSERT_TRUE(item_b_owner.has_value());
  EXPECT_EQ(item_a_owner.value(), 0U);
  EXPECT_EQ(item_b_owner.value(), 1U);
}

TEST_F(RocksDBStorageTest, InsertReplacingSameInternalIdCleansOldIndexes) {
  RocksDBConfig indexed_config = config_;
  indexed_config.indexed_fields_ = {"category"};
  RocksDBStorage<> storage(indexed_config);

  ASSERT_TRUE(
      storage.insert(7, ScalarData{"old_item", "old doc", {{"category", std::string("old")}}}));

  EXPECT_TRUE(
      storage.insert(7, ScalarData{"new_item", "new doc", {{"category", std::string("new")}}}));

  EXPECT_EQ(storage.count(), 1U);
  EXPECT_FALSE(storage.find_by_item_id("old_item").has_value());

  auto new_owner = storage.find_by_item_id("new_item");
  ASSERT_TRUE(new_owner.has_value());
  EXPECT_EQ(new_owner.value(), 7U);

  EXPECT_TRUE(storage.get_ids_by_field_value("category", std::string("old")).empty());
  auto new_category_ids = storage.get_ids_by_field_value("category", std::string("new"));
  ASSERT_EQ(new_category_ids.size(), 1U);
  EXPECT_EQ(new_category_ids[0], 7U);
}

TEST_F(RocksDBStorageTest, FieldIndexesSupportSimpleEqualityAndRangePredicates) {
  RocksDBConfig indexed_config = config_;
  indexed_config.indexed_fields_ = {"label", "price"};
  RocksDBStorage<> storage(indexed_config);

  ASSERT_TRUE(
      storage.insert(10,
                     ScalarData{"item_10",
                                "doc10",
                                {{"label", std::string("target")}, {"price", int64_t(1999)}}}));
  ASSERT_TRUE(
      storage.insert(2,
                     ScalarData{"item_02",
                                "doc02",
                                {{"label", std::string("target")}, {"price", int64_t(2500)}}}));
  ASSERT_TRUE(
      storage.insert(31,
                     ScalarData{"item_31",
                                "doc31",
                                {{"label", std::string("other")}, {"price", int64_t(1500)}}}));

  auto target_ids = storage.get_ids_by_field_value("label", std::string("target"));
  std::sort(target_ids.begin(), target_ids.end());
  EXPECT_EQ(target_ids, (std::vector<uint32_t>{2, 10}));

  auto cheap_ids =
      storage.get_ids_by_int_range("price", std::numeric_limits<int64_t>::min(), int64_t(1999));
  std::sort(cheap_ids.begin(), cheap_ids.end());
  EXPECT_EQ(cheap_ids, (std::vector<uint32_t>{10, 31}));
}

TEST_F(RocksDBStorageTest, RangeIndexCacheInvalidatesOnlyModifiedFields) {
  RocksDBConfig indexed_config = config_;
  indexed_config.indexed_fields_ = {"price", "rating"};
  RocksDBStorage<> storage(indexed_config);

  ASSERT_TRUE(storage.insert(0,
                             ScalarData{"item_0",
                                        "doc0",
                                        {{"price", int64_t(10)}, {"rating", int64_t(1)}}}));
  ASSERT_TRUE(storage.insert(1,
                             ScalarData{"item_1",
                                        "doc1",
                                        {{"price", int64_t(20)}, {"rating", int64_t(2)}}}));

  auto cheap_before = storage.get_int_range_blocked_bitset("price", 0, 15, 2);
  ASSERT_TRUE(cheap_before.has_value());
  EXPECT_EQ(cheap_before->matched_count_, 1U);
  EXPECT_FALSE(cheap_before->blocked_->get(0));
  EXPECT_TRUE(cheap_before->blocked_->get(1));

  auto low_rating_before = storage.get_int_range_blocked_bitset("rating", 0, 1, 2);
  ASSERT_TRUE(low_rating_before.has_value());
  EXPECT_EQ(low_rating_before->matched_count_, 1U);
  EXPECT_FALSE(low_rating_before->blocked_->get(0));
  EXPECT_TRUE(low_rating_before->blocked_->get(1));

  ASSERT_TRUE(storage.update(1,
                             ScalarData{"item_1",
                                        "doc1",
                                        {{"price", int64_t(20)}, {"rating", int64_t(0)}}}));

  auto cheap_after = storage.get_int_range_blocked_bitset("price", 0, 15, 2);
  ASSERT_TRUE(cheap_after.has_value());
  EXPECT_EQ(cheap_after->matched_count_, 1U);
  EXPECT_FALSE(cheap_after->blocked_->get(0));
  EXPECT_TRUE(cheap_after->blocked_->get(1));

  auto low_rating_after = storage.get_int_range_blocked_bitset("rating", 0, 1, 2);
  ASSERT_TRUE(low_rating_after.has_value());
  EXPECT_EQ(low_rating_after->matched_count_, 2U);
  EXPECT_FALSE(low_rating_after->blocked_->get(0));
  EXPECT_FALSE(low_rating_after->blocked_->get(1));
}

TEST_F(RocksDBStorageTest, RemoveOperations) {
  RocksDBStorage<> storage(config_);

  bool invalid_removed = storage.remove(999);
  EXPECT_FALSE(invalid_removed);

  ScalarData data1{"id_001", "doc1", {}};
  ScalarData data2{"id_002", "doc2", {}};
  ScalarData data3{"id_003", "doc3", {}};

  storage.insert(0, data1);
  storage.insert(1, data2);
  storage.insert(2, data3);

  EXPECT_EQ(storage.count(), 3U);

  bool result0 = storage.remove(0);
  EXPECT_TRUE(result0);
  EXPECT_EQ(storage.count(), 2U);
  EXPECT_FALSE(storage.is_valid(0));

  bool result1 = storage.remove(1);
  EXPECT_TRUE(result1);
  EXPECT_EQ(storage.count(), 1U);
  EXPECT_FALSE(storage.is_valid(1));

  bool result2 = storage.remove(2);
  EXPECT_TRUE(result2);
  EXPECT_EQ(storage.count(), 0U);
  EXPECT_FALSE(storage.is_valid(2));
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BatchInsert) {
  RocksDBStorage<> storage(config_);

  std::vector<ScalarData> inputs = {
      {"id_001", "doc1", {{"score", int64_t(100)}}},
      {"id_002", "doc2", {{"score", int64_t(200)}}},
      {"id_003", "doc3", {{"score", int64_t(300)}}},
  };

  bool success = storage.batch_insert(0, inputs.begin(), inputs.end());
  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 3U);

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto data = storage[i];
    EXPECT_EQ(data.item_id, inputs[i].item_id);
    EXPECT_EQ(data.document, inputs[i].document);
  }
}

TEST_F(RocksDBStorageTest, EmptyBatchInsert) {
  RocksDBStorage<> storage(config_);
  std::vector<ScalarData> empty_inputs;

  bool success = storage.batch_insert(0, empty_inputs.begin(), empty_inputs.end());
  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 0U);
}

TEST_F(RocksDBStorageTest, BatchInsertWithOffset) {
  RocksDBStorage<> storage(config_);

  std::vector<ScalarData> batch1 = {
      {"id_001", "doc1", {}},
      {"id_002", "doc2", {}},
  };

  std::vector<ScalarData> batch2 = {
      {"id_003", "doc3", {}},
      {"id_004", "doc4", {}},
  };

  storage.batch_insert(0, batch1.begin(), batch1.end());
  storage.batch_insert(2, batch2.begin(), batch2.end());

  EXPECT_EQ(storage.count(), 4U);
  EXPECT_EQ(storage[0].item_id, "id_001");
  EXPECT_EQ(storage[1].item_id, "id_002");
  EXPECT_EQ(storage[2].item_id, "id_003");
  EXPECT_EQ(storage[3].item_id, "id_004");
}

TEST_F(RocksDBStorageTest, BatchInsertRejectsDuplicateItemIdsBeforeWriting) {
  RocksDBStorage<> storage(config_);

  std::vector<ScalarData> duplicate_batch = {
      {"dup", "doc1", {}},
      {"dup", "doc2", {}},
  };

  EXPECT_FALSE(storage.batch_insert(0, duplicate_batch.begin(), duplicate_batch.end()));
  EXPECT_EQ(storage.count(), 0U);
  EXPECT_FALSE(storage.is_valid(0));
  EXPECT_FALSE(storage.is_valid(1));
  EXPECT_FALSE(storage.find_by_item_id("dup").has_value());
}

TEST_F(RocksDBStorageTest, BatchInsertRejectsExistingItemIdBeforeWritingAnyRecord) {
  RocksDBStorage<> storage(config_);

  ASSERT_TRUE(storage.insert(0, ScalarData{"taken", "existing", {}}));
  std::vector<ScalarData> batch = {
      {"fresh", "fresh doc", {}},
      {"taken", "duplicate doc", {}},
  };

  EXPECT_FALSE(storage.batch_insert(1, batch.begin(), batch.end()));
  EXPECT_EQ(storage.count(), 1U);
  EXPECT_FALSE(storage.is_valid(1));
  EXPECT_FALSE(storage.is_valid(2));
  EXPECT_FALSE(storage.find_by_item_id("fresh").has_value());

  auto taken_owner = storage.find_by_item_id("taken");
  ASSERT_TRUE(taken_owner.has_value());
  EXPECT_EQ(taken_owner.value(), 0U);
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_F(RocksDBStorageTest, ComplexMetadata) {
  RocksDBStorage<> storage(config_);

  MetadataMap meta;
  meta["int_field"] = int64_t(42);
  meta["float_field"] = 3.14;
  meta["string_field"] = std::string("hello");
  meta["bool_field"] = true;

  ScalarData data{"item_1", "content", meta};
  storage.insert(0, data);

  auto retrieved = storage[0];
  EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("int_field")), 42);
  EXPECT_FLOAT_EQ(std::get<double>(retrieved.metadata.at("float_field")), 3.14);
  EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("string_field")), "hello");
  EXPECT_EQ(std::get<bool>(retrieved.metadata.at("bool_field")), true);
}

TEST_F(RocksDBStorageTest, FindByItemId) {
  RocksDBStorage<> storage(config_);

  storage.insert(0, ScalarData{"user_001", "doc1", {}});
  storage.insert(1, ScalarData{"user_002", "doc2", {}});
  storage.insert(2, ScalarData{"user_003", "doc3", {}});

  auto id1 = storage.find_by_item_id("user_001");
  EXPECT_TRUE(id1.has_value());
  EXPECT_EQ(id1.value(), 0U);

  auto id2 = storage.find_by_item_id("user_002");
  EXPECT_TRUE(id2.has_value());
  EXPECT_EQ(id2.value(), 1U);

  auto id_notfound = storage.find_by_item_id("nonexistent");
  EXPECT_FALSE(id_notfound.has_value());
}

// ============================================================================
// Batch Get Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BatchGet) {
  RocksDBStorage<> storage(config_);

  std::vector<ScalarData> inputs = {
      {"id_001", "doc1", {}},
      {"id_002", "doc2", {}},
      {"id_003", "doc3", {}},
  };

  storage.batch_insert(0, inputs.begin(), inputs.end());

  std::vector<uint32_t> ids = {0, 1, 2};
  auto results = storage.batch_get(ids);

  EXPECT_EQ(results.size(), 3U);
  EXPECT_EQ(results[0].item_id, "id_001");
  EXPECT_EQ(results[1].item_id, "id_002");
  EXPECT_EQ(results[2].item_id, "id_003");
}

TEST_F(RocksDBStorageTest, ScanWithFilterReturnsNumericIdOrder) {
  RocksDBStorage<> storage(config_);

  ASSERT_TRUE(storage.insert(10, ScalarData{"id_010", "doc10", {}}));
  ASSERT_TRUE(storage.insert(9, ScalarData{"id_009", "doc9", {}}));

  auto results = storage.scan_with_filter([](const ScalarData &) {
    return true;
  });

  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(results[0].first, 9U);
  EXPECT_EQ(results[1].first, 10U);
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST_F(RocksDBStorageTest, PersistenceAcrossInstances) {
  {
    RocksDBStorage<> storage(config_);
    storage.insert(0, ScalarData{"id_001", "doc1", {{"key", std::string("value1")}}});
    storage.insert(1, ScalarData{"id_002", "doc2", {{"key", std::string("value2")}}});
    storage.flush();
  }

  {
    RocksDBStorage<> storage(config_);
    EXPECT_EQ(storage.count(), 2U);
    EXPECT_EQ(storage[0].item_id, "id_001");
    EXPECT_EQ(storage[1].item_id, "id_002");
  }
}

TEST_F(RocksDBStorageTest, LockConflictFallsBackToReadOnlyAndRejectsWrites) {
  RocksDBStorage<> primary(config_);
  ASSERT_TRUE(primary.insert(0, ScalarData{"item_1", "data1", {}}));

  RocksDBStorage<> secondary(config_);
  EXPECT_TRUE(secondary.is_read_only());
  EXPECT_TRUE(secondary.is_valid(0));
  EXPECT_EQ(secondary[0].item_id, "item_1");
  EXPECT_THROW(secondary.insert(1, ScalarData{"item_2", "data2", {}}), std::runtime_error);
}

TEST_F(RocksDBStorageTest, SaveCheckpointAndRestore) {
  fs::path checkpoint_path = temp_dir_ / "checkpoint";

  {
    RocksDBStorage<> storage(config_);
    storage.insert(0, ScalarData{"item_1", "data1", {}});
    storage.insert(1, ScalarData{"item_2", "data2", {}});
    storage.save(checkpoint_path.string());
  }

  EXPECT_TRUE(fs::exists(checkpoint_path / "CURRENT"));

  fs::path restored_path = temp_dir_ / "restored";
  fs::copy(checkpoint_path, restored_path, fs::copy_options::recursive);

  RocksDBConfig restore_config;
  restore_config.db_path_ = restored_path.string();
  RocksDBStorage<> restored_storage(restore_config);

  EXPECT_EQ(restored_storage.count(), 2U);
  EXPECT_EQ(restored_storage[0].item_id, "item_1");
  EXPECT_EQ(restored_storage[1].item_id, "item_2");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(RocksDBStorageTest, DatabaseOpenFailure) {
  RocksDBConfig bad_config;
  bad_config.db_path_ = "/invalid/nonexistent/path/to/db";
  bad_config.create_if_missing_ = false;

  EXPECT_THROW({ RocksDBStorage<> storage(bad_config); }, std::runtime_error);
}

TEST_F(RocksDBStorageTest, SaveCheckpointToInvalidPath) {
  RocksDBStorage<> storage(config_);
  storage.insert(0, ScalarData{"test", "data", {}});

  fs::path invalid_checkpoint_path = temp_dir_ / "nonexistent_parent/checkpoint";

  EXPECT_NO_THROW(storage.save(invalid_checkpoint_path.string()));
  EXPECT_FALSE(fs::exists(invalid_checkpoint_path));
}

// ============================================================================
// Configuration and Accessors Tests
// ============================================================================

TEST_F(RocksDBStorageTest, ConfigAndAccessors) {
  RocksDBStorage<> storage(config_);

  EXPECT_EQ(storage.get_db_path(), config_.db_path_);

  const auto &retrieved_config = storage.config();
  EXPECT_EQ(retrieved_config.db_path_, config_.db_path_);
  EXPECT_EQ(retrieved_config.write_buffer_size_, config_.write_buffer_size_);

  storage.insert(0, ScalarData{"test", "data", {}});
  EXPECT_NO_THROW(storage.get_statistics());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(RocksDBStorageTest, MoveSemantics) {
  {
    RocksDBStorage<> storage1(config_);
    storage1.insert(0, ScalarData{"test", "data", {}});

    RocksDBStorage<> storage2(std::move(storage1));
    EXPECT_EQ(storage2.count(), 1U);
    EXPECT_EQ(storage2[0].item_id, "test");
  }

  {
    RocksDBConfig config1 = config_;
    config1.db_path_ = (temp_dir_ / "db1").string();
    fs::create_directories(config1.db_path_);

    RocksDBConfig config2 = config_;
    config2.db_path_ = (temp_dir_ / "db2").string();
    fs::create_directories(config2.db_path_);

    RocksDBStorage<> storage1(config1);
    storage1.insert(0, ScalarData{"item1", "doc1", {}});

    RocksDBStorage<> storage2(config2);
    storage2.insert(0, ScalarData{"item2", "doc2", {}});

    storage2 = std::move(storage1);
    EXPECT_EQ(storage2.count(), 1U);
    EXPECT_EQ(storage2[0].item_id, "item1");
  }
}

// NOLINTEND
}  // namespace alaya
