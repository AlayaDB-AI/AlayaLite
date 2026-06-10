// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/diskann_index.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simd/distance_l2.hpp"

namespace {

#if defined(__linux__)

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// End-to-end fixture: builds a No-PQ index, loads it updatable, and mirrors the
// live (label -> vector) contents so recall can be measured against the actual
// current set after arbitrary insert/delete sequences.
class UpdateE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { spdlog::set_level(spdlog::level::warn); }

  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("diskann_upd_" + std::to_string(counter.fetch_add(1)));
  }
  void TearDown() override {
    idx_.reset();
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string dir() const { return dir_.string(); }

  void build_and_load(uint64_t n, uint64_t dim, uint32_t r, DiskANNLoadParams lp) {
    dim_ = dim;
    base_vecs_ = make_vectors(n, dim, /*seed=*/1);
    std::vector<uint64_t> labels(n);
    for (uint64_t i = 0; i < n; ++i) {
      labels[i] = 1000 + i;
    }
    DiskANNBuildParams bp;
    bp.R = r;
    bp.pq_n_chunks = 0;  // updates are No-PQ only
    DiskANNIndex::build(dir(), base_vecs_.data(), labels.data(), n, dim, bp);

    lp.updatable = true;
    idx_ = std::make_unique<DiskANNIndex>();
    idx_->load(dir(), lp);

    next_label_ = 1000 + n;
    medoid_ = idx_->medoid();
    live_.clear();
    id_label_.clear();
    deletable_.clear();
    for (uint64_t i = 0; i < n; ++i) {
      const uint64_t label = 1000 + i;
      live_[label] = vec_at(base_vecs_, i);
      id_label_[static_cast<uint32_t>(i)] = label;
      if (i != medoid_) {
        deletable_.push_back(static_cast<uint32_t>(i));
      }
    }
  }

  std::vector<float> vec_at(const std::vector<float> &flat, uint64_t i) const {
    return std::vector<float>(flat.begin() + i * dim_, flat.begin() + (i + 1) * dim_);
  }

  uint32_t do_insert(const std::vector<float> &v) {
    const uint64_t label = next_label_++;
    const uint32_t id = idx_->insert(v.data(), label);
    id_label_[id] = label;
    live_[label] = v;
    return id;
  }

  void do_remove(uint32_t id) {
    const auto it = id_label_.find(id);
    live_.erase(it->second);
    idx_->remove(id);
  }

  // Ground-truth top-k labels by exact L2 over the *current* live set.
  std::vector<uint64_t> ground_truth(const float *q, uint32_t k) const {
    std::vector<std::pair<float, uint64_t>> all;
    all.reserve(live_.size());
    for (const auto &[label, vec] : live_) {
      all.emplace_back(alaya::simd::l2_sqr<float, float>(q, vec.data(), dim_), label);
    }
    std::sort(all.begin(), all.end());
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < k && i < all.size(); ++i) {
      out.push_back(all[i].second);
    }
    return out;
  }

  double recall_at_k(const std::vector<float> &queries, uint32_t nq, uint32_t k, uint32_t l) const {
    uint64_t hits = 0;
    uint64_t total = 0;
    std::vector<uint64_t> out_l(k);
    std::vector<float> out_d(k);
    const DiskANNSearchParams sp{/*L=*/l,
                                 /*use_pq=*/false,
                                 /*rerank=*/false,
                                 /*rerank_count=*/0,
                                 /*deterministic=*/true};
    for (uint32_t qi = 0; qi < nq; ++qi) {
      const float *q = queries.data() + qi * dim_;
      const auto truth = ground_truth(q, k);
      idx_->search(q, k, out_l.data(), out_d.data(), sp);
      const std::unordered_set<uint64_t> res(out_l.begin(), out_l.end());
      for (const uint64_t t : truth) {
        if (res.count(t) != 0) {
          ++hits;
        }
        ++total;
      }
    }
    return total != 0 ? static_cast<double>(hits) / static_cast<double>(total) : 1.0;
  }

  bool top1_is(const float *q, uint64_t label) const {
    uint64_t out_l[5];
    float out_d[5];
    idx_->search(q,
                 5,
                 out_l,
                 out_d,
                 {/*L=*/64, /*use_pq=*/false, /*rerank=*/false, 0, /*deterministic=*/true});
    return out_l[0] == label;
  }

  std::filesystem::path dir_;
  uint64_t dim_ = 0;
  uint32_t medoid_ = 0;
  uint64_t next_label_ = 0;
  std::vector<float> base_vecs_;
  std::unique_ptr<DiskANNIndex> idx_;
  std::map<uint64_t, std::vector<float>> live_;      // label -> vector (live only)
  std::unordered_map<uint32_t, uint64_t> id_label_;  // internal id -> current label
  std::vector<uint32_t> deletable_;                  // live, non-medoid internal ids
};

// 7.1 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, InsertFindsNewVectorsWithCorrectLabels) {
  build_and_load(/*n=*/1000, /*dim=*/32, /*r=*/32, {});
  const auto nv = make_vectors(100, 32, /*seed=*/99);
  std::vector<uint64_t> labels;
  for (uint32_t i = 0; i < 100; ++i) {
    labels.push_back(0);
  }
  for (uint32_t i = 0; i < 100; ++i) {
    const uint32_t id = do_insert(vec_at(nv, i));
    labels[i] = id_label_[id];
  }
  EXPECT_EQ(idx_->live_count(), 1100u);

  uint32_t found = 0;
  for (uint32_t i = 0; i < 100; ++i) {
    if (top1_is(nv.data() + i * 32, labels[i])) {
      ++found;
    }
  }
  EXPECT_GE(found, 95u) << "inserted vectors should be findable as their own NN";
}

// 7.2 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, DeleteHidesVectorsAndPreservesRecall) {
  build_and_load(1000, 32, 32, {});
  const auto queries = make_vectors(50, 32, /*seed=*/7);

  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 100; ++i) {
    deleted.push_back(deletable_[i]);
  }
  for (const uint32_t id : deleted) {
    do_remove(id);
  }
  EXPECT_EQ(idx_->live_count(), 900u);
  EXPECT_EQ(idx_->tombstone_count(), 100u);

  for (const uint32_t id : {deleted[0], deleted[50], deleted[99]}) {
    uint64_t out_l[10];
    float out_d[10];
    idx_->search(base_vecs_.data() + id * 32, 10, out_l, out_d, {100, false, false, 0, true});
    for (const uint64_t l : out_l) {
      EXPECT_NE(l, static_cast<uint64_t>(1000 + id)) << "deleted vector must not be returned";
    }
  }
  EXPECT_GE(recall_at_k(queries, 50, 10, 100), 0.90);
}

// 7.3 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, MixedWorkloadKeepsRecallAbove90) {
  build_and_load(1000, 32, 32, {});
  const auto queries = make_vectors(50, 32, /*seed=*/7);
  const auto extra = make_vectors(400, 32, /*seed=*/555);  // 30 rounds * 10 inserts
  uint32_t extra_idx = 0;
  std::mt19937 rng(2024);

  for (int round = 0; round < 30; ++round) {
    for (int i = 0; i < 10; ++i) {  // 1% insert
      const uint32_t id = do_insert(vec_at(extra, extra_idx++));
      deletable_.push_back(id);
    }
    for (int i = 0; i < 10; ++i) {  // 1% delete (random live, non-medoid)
      std::uniform_int_distribution<size_t> pick(0, deletable_.size() - 1);
      const size_t j = pick(rng);
      const uint32_t id = deletable_[j];
      deletable_[j] = deletable_.back();
      deletable_.pop_back();
      do_remove(id);
    }
    const double r = recall_at_k(queries, 50, 10, 100);
    EXPECT_GE(r, 0.90) << "recall collapsed at round " << round;
  }
  EXPECT_EQ(idx_->live_count(), 1000u);  // balanced insert/delete
}

// 7.4 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, SlotReuseDrainsFreeListAndReoccupiesSlots) {
  build_and_load(/*n=*/200, /*dim=*/16, /*r=*/16, {});
  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 20; ++i) {
    deleted.push_back(deletable_[i]);
    do_remove(deletable_[i]);
  }
  EXPECT_EQ(idx_->free_slot_count(), 20u);
  const uint64_t cap_before = idx_->max_slot_id();

  const auto nv = make_vectors(20, 16, /*seed=*/321);
  std::unordered_set<uint32_t> reused;
  for (uint32_t i = 0; i < 20; ++i) {
    reused.insert(do_insert(vec_at(nv, i)));
  }
  EXPECT_EQ(idx_->free_slot_count(), 0u);      // free list drained
  EXPECT_EQ(idx_->max_slot_id(), cap_before);  // no append: capacity unchanged
  for (const uint32_t id : deleted) {
    EXPECT_TRUE(reused.count(id) != 0) << "freed slot " << id << " should have been reused";
  }
}

// 7.5 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, PersistenceAcrossFlushReload) {
  build_and_load(500, 32, 32, {});
  const auto nv = make_vectors(50, 32, /*seed=*/654);
  std::vector<uint64_t> inserted_labels;
  for (uint32_t i = 0; i < 50; ++i) {
    inserted_labels.push_back(id_label_[do_insert(vec_at(nv, i))]);
  }
  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 50; ++i) {
    deleted.push_back(deletable_[i]);
    do_remove(deletable_[i]);
  }
  const uint64_t live_before = idx_->live_count();
  const uint64_t tomb_before = idx_->tombstone_count();

  idx_->flush();
  idx_ = std::make_unique<DiskANNIndex>();
  DiskANNLoadParams lp;
  lp.updatable = true;
  idx_->load(dir(), lp);

  EXPECT_EQ(idx_->live_count(), live_before);
  EXPECT_EQ(idx_->tombstone_count(), tomb_before);
  for (const uint32_t id : deleted) {
    EXPECT_TRUE(idx_->is_deleted(id)) << "tombstone for " << id << " must survive reload";
  }
  for (const uint32_t id : {deleted[0], deleted[25], deleted[49]}) {
    uint64_t out_l[10];
    float out_d[10];
    idx_->search(base_vecs_.data() + id * 32, 10, out_l, out_d, {100, false, false, 0, true});
    for (const uint64_t l : out_l) {
      EXPECT_NE(l, static_cast<uint64_t>(1000 + id));
    }
  }
  uint32_t found = 0;
  for (uint32_t i = 0; i < 50; ++i) {
    if (top1_is(nv.data() + i * 32, inserted_labels[i])) {
      ++found;
    }
  }
  EXPECT_GE(found, 45u) << "inserted vectors must remain findable after reload";
}

// 7.6 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, SafetyNetReconnectFiresOnPureDeletes) {
  DiskANNLoadParams lp;
  lp.safety_net_ratio = 0.05;
  lp.safety_net_ops = 8;
  build_and_load(1000, 32, 32, lp);
  const auto queries = make_vectors(50, 32, /*seed=*/7);

  for (uint32_t i = 0; i < 80; ++i) {  // 8% pure deletes (exceeds 5% threshold)
    do_remove(deletable_[i]);
  }
  EXPECT_GT(idx_->safety_net_fire_count(), 0u) << "proactive reconnect should fire";
  EXPECT_GE(recall_at_k(queries, 50, 10, 100), 0.85) << "recall must not collapse";
}

#else  // !__linux__

TEST(UpdateE2ETest, SkippedOnNonLinux) {
  GTEST_SKIP() << "DiskANN in-place updates require Linux O_DIRECT";
}

#endif  // __linux__

}  // namespace
