// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/diskann_index.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coro/thread_pool.hpp"
#include "simd/distance_l2.hpp"

#if defined(__linux__)
  #include <unistd.h>
#endif

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

// End-to-end fixture: builds an index, loads it updatable, and mirrors the
// live (label -> vector) contents so recall can be measured against the actual
// current set after arbitrary insert/delete sequences.
class UpdateE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { spdlog::set_level(spdlog::level::warn); }

  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() / ("diskann_upd_" + std::to_string(::getpid()) +
                                                     "_" + std::to_string(counter.fetch_add(1)));
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  void TearDown() override {
    idx_.reset();
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string dir() const { return dir_.string(); }

  void build_and_load(uint64_t n,
                      uint64_t dim,
                      uint32_t r,
                      DiskANNLoadParams lp,
                      uint32_t pq_n_chunks = 0) {
    dim_ = dim;
    base_vecs_ = make_vectors(n, dim, /*seed=*/1);
    std::vector<uint64_t> labels(n);
    for (uint64_t i = 0; i < n; ++i) {
      labels[i] = 1000 + i;
    }
    DiskANNBuildParams bp;
    bp.R = r;
    bp.pq_n_chunks = pq_n_chunks;
    DiskANNIndex::build(dir(), base_vecs_.data(), labels.data(), n, dim, bp);

    lp.updatable = true;
    // ALAYA_DISKANN_UPDATE_IO=blocking|uring|auto lets CI exercise both update
    // I/O backends with the same suite (default auto = uring when available).
    if (const char *mode = std::getenv("ALAYA_DISKANN_UPDATE_IO")) {
      if (std::string_view(mode) == "blocking") {
        lp.update_io = alaya::diskann::DiskANNUpdateIO::kBlocking;
      } else if (std::string_view(mode) == "uring") {
        lp.update_io = alaya::diskann::DiskANNUpdateIO::kUring;
      }
    }
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

  bool topk_contains(const float *q,
                     uint32_t k,
                     uint64_t label,
                     const DiskANNSearchParams &sp) const {
    std::vector<uint64_t> out_l(k);
    std::vector<float> out_d(k);
    idx_->search(q, k, out_l.data(), out_d.data(), sp);
    return std::find(out_l.begin(), out_l.end(), label) != out_l.end();
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
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, {});
  const auto nv = make_vectors(40, 32, /*seed=*/99);
  std::vector<uint64_t> labels;
  for (uint32_t i = 0; i < 40; ++i) {
    labels.push_back(0);
  }
  for (uint32_t i = 0; i < 40; ++i) {
    const uint32_t id = do_insert(vec_at(nv, i));
    labels[i] = id_label_[id];
  }
  EXPECT_EQ(idx_->live_count(), 540u);

  uint32_t found = 0;
  for (uint32_t i = 0; i < 40; ++i) {
    if (top1_is(nv.data() + i * 32, labels[i])) {
      ++found;
    }
  }
  EXPECT_GE(found, 38u) << "inserted vectors should be findable as their own NN";
}

TEST_F(UpdateE2ETest, InsertWithPageCacheIsVisibleToSearchAfterReturn) {
  DiskANNLoadParams lp;
  lp.page_cache_capacity = 8;
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, lp);

  const auto nv = make_vectors(1, 32, /*seed=*/991);
  const uint32_t id = do_insert(vec_at(nv, 0));
  ASSERT_TRUE(top1_is(nv.data(), id_label_[id]));
}

TEST_F(UpdateE2ETest, ExternalLabelsSupportLookupDeleteAndReuse) {
  build_and_load(/*n=*/300, /*dim=*/32, /*r=*/32, {});
  const uint32_t internal_id = deletable_.front();
  const uint64_t label = id_label_[internal_id];

  EXPECT_TRUE(idx_->contains_label(label));
  idx_->remove_by_label(label);
  EXPECT_FALSE(idx_->contains_label(label));
  EXPECT_TRUE(idx_->is_deleted(internal_id));
  EXPECT_THROW(idx_->remove_by_label(label), std::out_of_range);

  const auto replacement = make_vectors(1, 32, /*seed=*/8128);
  const uint32_t reused_id = idx_->insert(replacement.data(), label);
  EXPECT_EQ(reused_id, internal_id);
  EXPECT_TRUE(idx_->contains_label(label));
  EXPECT_TRUE(top1_is(replacement.data(), label));
}

TEST_F(UpdateE2ETest, InsertRejectsDuplicateExternalLabelsBeforeMutation) {
  build_and_load(/*n=*/300, /*dim=*/32, /*r=*/32, {});
  const auto vectors = make_vectors(2, 32, /*seed=*/2048);
  const uint64_t live_before = idx_->live_count();

  EXPECT_THROW(idx_->insert(vectors.data(), 1001), std::invalid_argument);
  EXPECT_EQ(idx_->live_count(), live_before);

  const uint64_t labels[] = {9000, 9000};
  EXPECT_THROW(idx_->batch_insert(vectors.data(), labels, 2, 2), std::invalid_argument);
  EXPECT_EQ(idx_->live_count(), live_before);
  EXPECT_FALSE(idx_->contains_label(9000));
}

TEST_F(UpdateE2ETest, InsertWithParallelReconnectKeepsVectorsSearchable) {
  DiskANNLoadParams lp;
  lp.page_cache_capacity = 16;
  lp.update_reconnect_threads = 4;
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, lp);

  const auto nv = make_vectors(8, 32, /*seed=*/4411);
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t id = do_insert(vec_at(nv, i));
    ASSERT_TRUE(top1_is(nv.data() + static_cast<uint64_t>(i) * 32, id_label_[id]))
        << "inserted vector " << i << " should be searchable after parallel reconnect";
  }
}

TEST_F(UpdateE2ETest, RecordCapacityLeavesUpdateHeadroom) {
  // Yi-style layout: records hold more neighbor slots than the built graph
  // degree, so insert reconnects keep pools verbatim instead of pruning.
  dim_ = 32;
  base_vecs_ = make_vectors(300, 32, /*seed=*/7);
  std::vector<uint64_t> labels(300);
  for (uint64_t i = 0; i < labels.size(); ++i) {
    labels[i] = 1000 + i;
  }
  DiskANNBuildParams bp;
  bp.R = 16;
  bp.record_capacity = 24;
  bp.pq_n_chunks = 0;
  DiskANNIndex::build(dir(), base_vecs_.data(), labels.data(), labels.size(), 32, bp);

  DiskANNLoadParams lp;
  lp.updatable = true;
  idx_ = std::make_unique<DiskANNIndex>();
  idx_->load(dir(), lp);
  next_label_ = 1000 + 300;
  medoid_ = idx_->medoid();
  live_.clear();
  id_label_.clear();
  for (uint64_t i = 0; i < 300; ++i) {
    live_[1000 + i] = vec_at(base_vecs_, i);
    id_label_[static_cast<uint32_t>(i)] = 1000 + i;
  }

  const auto nv = make_vectors(8, 32, /*seed=*/77);
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t id = do_insert(vec_at(nv, i));
    ASSERT_TRUE(top1_is(nv.data() + static_cast<uint64_t>(i) * 32, id_label_[id]))
        << "insert " << i << " should be findable on a capacity-padded layout";
  }
  idx_->flush();

  // Reload and confirm the capacity survives the meta round-trip.
  DiskANNIndex reloaded;
  reloaded.load(dir(), lp);
  const auto q = vec_at(nv, 0);
  uint64_t out_l = 0;
  float out_d = 0.0f;
  DiskANNSearchParams sp;
  sp.use_pq = false;
  sp.rerank = false;
  reloaded.search(q.data(), 1, &out_l, &out_d, sp);
  EXPECT_EQ(out_l, 1300u);  // first insert's label (next_label_ started at 1300)
}

TEST_F(UpdateE2ETest, BuildRejectsCapacityBelowR) {
  const auto vecs = make_vectors(100, 16, /*seed=*/23);
  std::vector<uint64_t> labels(100);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = i;
  }
  DiskANNBuildParams bp;
  bp.R = 16;
  bp.record_capacity = 8;
  bp.pq_n_chunks = 0;
  EXPECT_THROW(DiskANNIndex::build(dir(), vecs.data(), labels.data(), 100, 16, bp),
               std::invalid_argument);
}

TEST_F(UpdateE2ETest, LoadRejectsZeroUpdateReconnectThreads) {
  const auto vecs = make_vectors(100, 16, /*seed=*/19);
  std::vector<uint64_t> labels(100);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 1000 + i;
  }
  DiskANNBuildParams bp;
  bp.R = 16;
  bp.pq_n_chunks = 0;
  DiskANNIndex::build(dir(), vecs.data(), labels.data(), labels.size(), 16, bp);

  DiskANNLoadParams lp;
  lp.updatable = true;
  lp.update_reconnect_threads = 0;
  DiskANNIndex idx;
  EXPECT_THROW(idx.load(dir(), lp), std::invalid_argument);
}

TEST_F(UpdateE2ETest, BatchInsertUsesConfiguredUpdateWorkersWithSingleSearchThread) {
  DiskANNLoadParams lp;
  lp.num_threads = 1;
  lp.update_insert_threads = 8;
  lp.update_reconnect_threads = 4;
  lp.page_cache_capacity = 16;
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, lp);

  const auto nv = make_vectors(24, 32, /*seed=*/7070);
  std::vector<uint64_t> labels(24);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 12000 + i;
  }

  const std::vector<uint32_t> ids =
      idx_->batch_insert(nv.data(), labels.data(), static_cast<uint32_t>(labels.size()), 8);

  ASSERT_EQ(ids.size(), labels.size());
  EXPECT_EQ(idx_->live_count(), 524u);
  for (uint32_t i = 0; i < ids.size(); ++i) {
    id_label_[ids[i]] = labels[i];
    live_[labels[i]] = vec_at(nv, i);
    EXPECT_TRUE(top1_is(nv.data() + static_cast<uint64_t>(i) * 32, labels[i]))
        << "batch inserted vector " << i << " should be searchable";
  }
}

TEST_F(UpdateE2ETest, BatchInsertReturnsIdsAndMakesVectorsSearchable) {
  DiskANNLoadParams lp;
  lp.page_cache_capacity = 16;
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, lp);

  const auto nv = make_vectors(16, 32, /*seed=*/12345);
  std::vector<uint64_t> labels(16);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 9000 + i;
  }

  const std::vector<uint32_t> ids =
      idx_->batch_insert(nv.data(), labels.data(), static_cast<uint32_t>(labels.size()), 8);

  ASSERT_EQ(ids.size(), labels.size());
  EXPECT_EQ(idx_->live_count(), 516u);
  for (uint32_t i = 0; i < ids.size(); ++i) {
    id_label_[ids[i]] = labels[i];
    live_[labels[i]] = vec_at(nv, i);
    EXPECT_TRUE(top1_is(nv.data() + static_cast<uint64_t>(i) * 32, labels[i]))
        << "batch inserted vector " << i << " should be searchable";
  }
}

TEST_F(UpdateE2ETest, ExternalPoolBatchUpdateKeepsVectorsSearchable) {
  DiskANNLoadParams lp;
  lp.num_threads = 2;
  lp.page_cache_capacity = 16;
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, lp);

  std::vector<uint32_t> removed;
  removed.assign(deletable_.begin(), deletable_.begin() + 4);
  coro::thread_pool pool{
      {.thread_count = 4, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  idx_->batch_remove_with_pool(removed.data(), static_cast<uint32_t>(removed.size()), pool);
  for (const uint32_t id : removed) {
    live_.erase(id_label_[id]);
  }

  const auto nv = make_vectors(4, 32, /*seed=*/5151);
  std::vector<uint64_t> labels(4);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 15000 + i;
  }
  const std::vector<uint32_t> ids =
      idx_->batch_insert_with_pool(nv.data(),
                                   labels.data(),
                                   static_cast<uint32_t>(labels.size()),
                                   4,
                                   pool);
  pool.shutdown();

  ASSERT_EQ(ids.size(), labels.size());
  EXPECT_EQ(idx_->live_count(), 500u);
  for (uint32_t i = 0; i < ids.size(); ++i) {
    id_label_[ids[i]] = labels[i];
    live_[labels[i]] = vec_at(nv, i);
    EXPECT_TRUE(top1_is(nv.data() + static_cast<uint64_t>(i) * 32, labels[i]))
        << "external-pool batch inserted vector " << i << " should be searchable";
  }
}

TEST_F(UpdateE2ETest, PQInsertUpdatesCodesAndPersistsAcrossReload) {
  build_and_load(/*n=*/300, /*dim=*/32, /*r=*/32, {}, /*pq_n_chunks=*/8);
  ASSERT_TRUE(idx_->has_pq());

  const auto nv = make_vectors(1, 32, /*seed=*/8181);
  const uint32_t id = do_insert(vec_at(nv, 0));
  const uint64_t label = id_label_[id];
  const DiskANNSearchParams pq_search{/*L=*/96,
                                      /*use_pq=*/true,
                                      /*rerank=*/true,
                                      /*rerank_count=*/0,
                                      /*deterministic=*/true};
  ASSERT_TRUE(topk_contains(nv.data(), /*k=*/10, label, pq_search));

  idx_->flush();
  const auto pq_codes_size = std::filesystem::file_size(dir_ / "pq_compressed.bin");
  EXPECT_EQ(pq_codes_size, idx_->max_slot_id() * 8u);

  idx_ = std::make_unique<DiskANNIndex>();
  DiskANNLoadParams lp;
  lp.updatable = true;
  idx_->load(dir(), lp);

  EXPECT_TRUE(idx_->has_pq());
  EXPECT_EQ(idx_->live_count(), 301u);
  EXPECT_TRUE(topk_contains(nv.data(), /*k=*/10, label, pq_search));
}

TEST_F(UpdateE2ETest, PQSearchSkipsDeletedLabels) {
  build_and_load(/*n=*/300, /*dim=*/32, /*r=*/32, {}, /*pq_n_chunks=*/8);

  const uint32_t deleted_id = deletable_[0];
  const uint64_t deleted_label = id_label_[deleted_id];
  do_remove(deleted_id);

  uint64_t out_l[10];
  float out_d[10];
  const DiskANNSearchParams pq_search{/*L=*/96,
                                      /*use_pq=*/true,
                                      /*rerank=*/true,
                                      /*rerank_count=*/0,
                                      /*deterministic=*/true};
  idx_->search(base_vecs_.data() + static_cast<uint64_t>(deleted_id) * 32,
               10,
               out_l,
               out_d,
               pq_search);
  for (const uint64_t label : out_l) {
    EXPECT_NE(label, deleted_label);
  }
}

TEST_F(UpdateE2ETest, PipelinedSearchMatchesBatchSearch) {
  build_and_load(/*n=*/600, /*dim=*/32, /*r=*/32, {}, /*pq_n_chunks=*/8);
  ASSERT_TRUE(idx_->has_pq());

  // Post-update state: labels beyond the base range plus live tombstones.
  const auto extra = make_vectors(20, 32, /*seed=*/4242);
  for (uint32_t i = 0; i < 20; ++i) {
    do_insert(vec_at(extra, i));
  }
  for (uint32_t i = 0; i < 10; ++i) {
    do_remove(deletable_[i]);
  }

  constexpr uint32_t kNq = 60;
  constexpr uint32_t kK = 5;
  const auto queries = make_vectors(kNq, 32, /*seed=*/7);
  // The pipelined path reproduces the sync rerank semantics without rerank
  // reads: it takes the full exact-scored search list and cuts top_k by exact
  // distance (traversal expands every retset entry, so rerank re-reads 0
  // nodes). Reference = sync search WITH rerank over the same L-sized pool;
  // the pipelined call itself rejects rerank=true.
  const DiskANNSearchParams ref_sp{/*L=*/48,
                                   /*use_pq=*/true,
                                   /*rerank=*/true,
                                   /*rerank_count=*/48,
                                   /*deterministic=*/false};
  const DiskANNSearchParams sp{/*L=*/48,
                               /*use_pq=*/true,
                               /*rerank=*/false,
                               /*rerank_count=*/0,
                               /*deterministic=*/false};

  std::vector<uint64_t> ref_l(kNq * kK);
  std::vector<float> ref_d(kNq * kK);
  idx_->batch_search(
      queries.data(), kNq, kK, ref_l.data(), ref_d.data(), /*num_threads=*/2, ref_sp);

  std::vector<uint64_t> pipe_l(kNq * kK, 0);
  std::vector<float> pipe_d(kNq * kK, 0.0F);
  const char *mode = std::getenv("ALAYA_DISKANN_UPDATE_IO");
  if (mode != nullptr && std::string_view(mode) == "blocking") {
    // No reactor in blocking mode: the pipelined path must refuse loudly.
    EXPECT_THROW(idx_->search_pipelined(
                     queries.data(), kNq, kK, pipe_l.data(), pipe_d.data(), 2, 8, sp),
                 std::runtime_error);
    return;
  }
  idx_->search_pipelined(queries.data(),
                         kNq,
                         kK,
                         pipe_l.data(),
                         pipe_d.data(),
                         /*num_threads=*/2,
                         /*pipeline=*/8,
                         sp);

  uint64_t ref_hits = 0;
  uint64_t pipe_hits = 0;
  uint64_t total = 0;
  for (uint32_t qi = 0; qi < kNq; ++qi) {
    const float *q = queries.data() + static_cast<uint64_t>(qi) * 32;
    const auto truth = ground_truth(q, kK);
    std::unordered_set<uint64_t> ref_set;
    std::unordered_set<uint64_t> pipe_set;
    for (uint32_t i = 0; i < kK; ++i) {
      const uint64_t rl = ref_l[qi * kK + i];
      const uint64_t pl = pipe_l[qi * kK + i];
      if (rl != DiskANNIndex::kNoLabel) {
        ref_set.insert(rl);
      }
      if (pl != DiskANNIndex::kNoLabel) {
        pipe_set.insert(pl);
        // Every pipelined hit must be a live label and carry its exact L2
        // distance (validates label mapping + the distance channel).
        const auto it = live_.find(pl);
        ASSERT_NE(it, live_.end()) << "pipelined search returned a dead/unknown label";
        const float exact = alaya::simd::l2_sqr<float, float>(q, it->second.data(), 32);
        EXPECT_NEAR(pipe_d[qi * kK + i], exact, 1e-3F * std::max(1.0F, exact));
      }
    }
    EXPECT_FALSE(pipe_set.empty()) << "query " << qi << " returned no results";
    for (const uint64_t t : truth) {
      ++total;
      ref_hits += ref_set.count(t);
      pipe_hits += pipe_set.count(t);
    }
  }
  const double ref_recall = static_cast<double>(ref_hits) / static_cast<double>(total);
  const double pipe_recall = static_cast<double>(pipe_hits) / static_cast<double>(total);
  // Same recall-equivalence class as the sync schedulers: parity within noise.
  // No high absolute bar — rerank-free PQ over random uniform vectors tops out
  // around ~0.75 recall for BOTH paths; parity is the invariant under test.
  EXPECT_NEAR(pipe_recall, ref_recall, 0.03) << "pipe=" << pipe_recall << " ref=" << ref_recall;
  EXPECT_GE(pipe_recall, 0.60);
}

// 7.2 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, DeleteHidesVectorsAndPreservesRecall) {
  build_and_load(500, 32, 32, {});
  const auto queries = make_vectors(20, 32, /*seed=*/7);

  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 50; ++i) {
    deleted.push_back(deletable_[i]);
  }
  for (const uint32_t id : deleted) {
    do_remove(id);
  }
  EXPECT_EQ(idx_->live_count(), 450u);
  EXPECT_EQ(idx_->tombstone_count(), 50u);

  for (const uint32_t id : {deleted[0], deleted[25], deleted[49]}) {
    uint64_t out_l[10];
    float out_d[10];
    idx_->search(base_vecs_.data() + id * 32, 10, out_l, out_d, {100, false, false, 0, true});
    for (const uint64_t l : out_l) {
      EXPECT_NE(l, static_cast<uint64_t>(1000 + id)) << "deleted vector must not be returned";
    }
  }
  EXPECT_GE(recall_at_k(queries, 20, 10, 100), 0.90);
}

// 7.3 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, MixedWorkloadKeepsRecallAbove90) {
  build_and_load(500, 32, 32, {});
  const auto queries = make_vectors(20, 32, /*seed=*/7);
  const auto extra = make_vectors(80, 32, /*seed=*/555);
  uint32_t extra_idx = 0;
  std::mt19937 rng(2024);

  for (int round = 0; round < 10; ++round) {
    for (int i = 0; i < 5; ++i) {
      const uint32_t id = do_insert(vec_at(extra, extra_idx++));
      deletable_.push_back(id);
    }
    for (int i = 0; i < 5; ++i) {
      std::uniform_int_distribution<size_t> pick(0, deletable_.size() - 1);
      const size_t j = pick(rng);
      const uint32_t id = deletable_[j];
      deletable_[j] = deletable_.back();
      deletable_.pop_back();
      do_remove(id);
    }
    const double r = recall_at_k(queries, 20, 10, 100);
    EXPECT_GE(r, 0.90) << "recall collapsed at round " << round;
  }
  EXPECT_EQ(idx_->live_count(), 500u);  // balanced insert/delete
}

TEST_F(UpdateE2ETest, SearchReturnsWhileBatchInsertIsInFlight) {
  DiskANNLoadParams lp;
  lp.num_threads = 2;
  lp.update_insert_threads = 1;
  lp.update_reconnect_threads = 1;
  build_and_load(/*n=*/700, /*dim=*/32, /*r=*/32, lp);

  const uint32_t n_insert = 96;
  const auto nv = make_vectors(n_insert, 32, /*seed=*/9876);
  std::vector<uint64_t> labels(n_insert);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 20000 + i;
  }

  std::atomic<bool> insert_started{false};
  auto insert_future = std::async(std::launch::async, [&]() {
    insert_started.store(true, std::memory_order_release);
    return idx_->batch_insert(nv.data(), labels.data(), n_insert, /*batch_size=*/1);
  });
  while (!insert_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(insert_future.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout)
      << "batch_insert must still be running for the concurrency assertion";

  auto search_future = std::async(std::launch::async, [&]() {
    uint64_t out_l[10];
    float out_d[10];
    idx_->search(base_vecs_.data(), 10, out_l, out_d, {/*L=*/100, false, false, 0, true});
  });

  EXPECT_EQ(search_future.wait_for(std::chrono::milliseconds(50)), std::future_status::ready)
      << "search should run while a batch insert is in flight";
  EXPECT_EQ(insert_future.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout)
      << "batch_insert should not have completed before the overlapping search returns";

  search_future.get();
  const std::vector<uint32_t> ids = insert_future.get();
  ASSERT_EQ(ids.size(), labels.size());
}

TEST_F(UpdateE2ETest, ReusedSlotStaysTombstonedUntilBatchInsertPublishesIt) {
  DiskANNLoadParams lp;
  lp.num_threads = 2;
  lp.update_insert_threads = 1;
  lp.update_reconnect_threads = 1;
  build_and_load(/*n=*/700, /*dim=*/32, /*r=*/32, lp);

  const uint32_t deleted_id = deletable_[0];
  do_remove(deleted_id);
  ASSERT_TRUE(idx_->is_deleted(deleted_id));

  const uint32_t n_insert = 96;
  const auto nv = make_vectors(n_insert, 32, /*seed=*/8765);
  std::vector<uint64_t> labels(n_insert);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    labels[i] = 30000 + i;
  }

  std::atomic<bool> insert_started{false};
  auto insert_future = std::async(std::launch::async, [&]() {
    insert_started.store(true, std::memory_order_release);
    return idx_->batch_insert(nv.data(), labels.data(), n_insert, /*batch_size=*/1);
  });
  while (!insert_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  // The tombstone is monotone for this slot while the batch runs: only the
  // first sub-batch's publish clears it and nothing re-sets it. That makes
  // two invariants probeable without assuming how fast the first insert
  // publishes (a fixed sleep here loses once inserts get fast enough). The
  // slot state is re-read AFTER each search so a publish landing mid-probe
  // cannot fake a violation:
  //   1. while tombstoned, the slot's new label must stay unsearchable —
  //      a dark record is masked until written and published;
  //   2. the tombstone never reappears once cleared.
  bool was_live = false;
  const DiskANNSearchParams probe_sp{/*L=*/64,
                                     /*use_pq=*/false,
                                     /*rerank=*/false,
                                     /*rerank_count=*/0,
                                     /*deterministic=*/true};
  while (insert_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout) {
    const bool found = topk_contains(nv.data(), 5, labels[0], probe_sp);
    if (idx_->is_deleted(deleted_id)) {
      EXPECT_FALSE(was_live) << "tombstone reappeared after the reused slot went live";
      EXPECT_FALSE(found)
          << "a reused slot must remain tombstoned until its new record is written and published";
    } else {
      was_live = true;
    }
    std::this_thread::yield();
  }

  const std::vector<uint32_t> ids = insert_future.get();
  ASSERT_EQ(ids.size(), labels.size());
  EXPECT_EQ(ids.front(), deleted_id);
  EXPECT_FALSE(idx_->is_deleted(deleted_id));
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

TEST_F(UpdateE2ETest, BatchRemoveHidesVectorsAndFeedsFreeList) {
  build_and_load(/*n=*/500, /*dim=*/32, /*r=*/32, {});
  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 40; ++i) {
    deleted.push_back(deletable_[i]);
    live_.erase(id_label_[deletable_[i]]);
  }

  idx_->batch_remove(deleted.data(), static_cast<uint32_t>(deleted.size()));

  EXPECT_EQ(idx_->live_count(), 460u);
  EXPECT_EQ(idx_->tombstone_count(), 40u);
  EXPECT_EQ(idx_->free_slot_count(), 40u);
  for (const uint32_t id : deleted) {
    EXPECT_TRUE(idx_->is_deleted(id));
  }

  const auto queries = make_vectors(30, 32, /*seed=*/7);
  EXPECT_GE(recall_at_k(queries, 30, 10, 100), 0.85);

  const uint64_t cap_before = idx_->max_slot_id();
  const auto nv = make_vectors(40, 32, /*seed=*/765);
  std::unordered_set<uint32_t> reused;
  for (uint32_t i = 0; i < 40; ++i) {
    reused.insert(do_insert(vec_at(nv, i)));
  }
  EXPECT_EQ(idx_->max_slot_id(), cap_before);
  for (const uint32_t id : deleted) {
    EXPECT_TRUE(reused.count(id) != 0);
  }
}

TEST_F(UpdateE2ETest, BatchRemoveByLabelsValidatesBeforeMutation) {
  build_and_load(/*n=*/300, /*dim=*/32, /*r=*/32, {});
  const uint64_t first_label = id_label_[deletable_[0]];
  const uint64_t second_label = id_label_[deletable_[1]];
  const uint64_t live_before = idx_->live_count();
  const uint64_t invalid_labels[] = {first_label, 999999};

  EXPECT_THROW(idx_->batch_remove_by_labels(invalid_labels, 2), std::out_of_range);
  EXPECT_EQ(idx_->live_count(), live_before);
  EXPECT_TRUE(idx_->contains_label(first_label));

  const uint64_t labels[] = {first_label, second_label};
  idx_->batch_remove_by_labels(labels, 2);
  EXPECT_EQ(idx_->live_count(), live_before - 2);
  EXPECT_FALSE(idx_->contains_label(first_label));
  EXPECT_FALSE(idx_->contains_label(second_label));
}

// 7.5 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, PersistenceAcrossFlushReload) {
  build_and_load(300, 32, 32, {});
  const auto nv = make_vectors(24, 32, /*seed=*/654);
  std::vector<uint64_t> inserted_labels;
  for (uint32_t i = 0; i < 24; ++i) {
    inserted_labels.push_back(id_label_[do_insert(vec_at(nv, i))]);
  }
  std::vector<uint32_t> deleted;
  for (uint32_t i = 0; i < 24; ++i) {
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
  for (const uint64_t label : inserted_labels) {
    EXPECT_TRUE(idx_->contains_label(label));
  }
  for (const uint32_t id : deleted) {
    EXPECT_TRUE(idx_->is_deleted(id)) << "tombstone for " << id << " must survive reload";
    EXPECT_FALSE(idx_->contains_label(id_label_[id]));
  }
  for (const uint32_t id : {deleted[0], deleted[12], deleted[23]}) {
    uint64_t out_l[10];
    float out_d[10];
    idx_->search(base_vecs_.data() + id * 32, 10, out_l, out_d, {100, false, false, 0, true});
    for (const uint64_t l : out_l) {
      EXPECT_NE(l, static_cast<uint64_t>(1000 + id));
    }
  }
  uint32_t found = 0;
  for (uint32_t i = 0; i < 24; ++i) {
    if (top1_is(nv.data() + i * 32, inserted_labels[i])) {
      ++found;
    }
  }
  EXPECT_GE(found, 20u) << "inserted vectors must remain findable after reload";
}

// 7.6 -----------------------------------------------------------------------
TEST_F(UpdateE2ETest, SafetyNetReconnectFiresOnPureDeletes) {
  DiskANNLoadParams lp;
  lp.safety_net_ratio = 0.05;
  lp.safety_net_ops = 8;
  build_and_load(500, 32, 32, lp);
  const auto queries = make_vectors(20, 32, /*seed=*/7);

  for (uint32_t i = 0; i < 40; ++i) {  // 8% pure deletes (exceeds 5% threshold)
    do_remove(deletable_[i]);
  }
  EXPECT_GT(idx_->safety_net_fire_count(), 0u) << "proactive reconnect should fire";
  EXPECT_GE(recall_at_k(queries, 20, 10, 100), 0.85) << "recall must not collapse";
}

#else  // !__linux__

TEST(UpdateE2ETest, SkippedOnNonLinux) {
  GTEST_SKIP() << "DiskANN in-place updates require Linux O_DIRECT";
}

#endif  // __linux__

}  // namespace
