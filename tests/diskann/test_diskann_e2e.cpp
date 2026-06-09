// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// End-to-end DiskANN index tests: build a 10K-vector index on disk, then load
// and search it, measuring recall against brute-force ground truth.

#include "index/graph/diskann/diskann_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "simd/distance_l2.hpp"

namespace {

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;

constexpr uint64_t kN = 10000;
constexpr uint64_t kDim = 128;
constexpr uint32_t kNQ = 100;
constexpr uint32_t kTopK = 10;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// Exact top-k internal ids per query (brute force ground truth).
std::vector<std::vector<uint32_t>> ground_truth(const std::vector<float> &data, uint64_t n,
                                                uint64_t dim, const std::vector<float> &queries,
                                                uint32_t nq, uint32_t k) {
  std::vector<std::vector<uint32_t>> gt(nq);
  for (uint32_t q = 0; q < nq; ++q) {
    const float *query = queries.data() + static_cast<uint64_t>(q) * dim;
    std::vector<std::pair<float, uint32_t>> d;
    d.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      d.emplace_back(alaya::simd::l2_sqr<float, float>(query, data.data() + i * dim, dim),
                     static_cast<uint32_t>(i));
    }
    std::partial_sort(d.begin(), d.begin() + k, d.end());
    gt[q].reserve(k);
    for (uint32_t i = 0; i < k; ++i) {
      gt[q].push_back(d[i].second);
    }
  }
  return gt;
}

class DiskANNE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    data_ = make_vectors(kN, kDim, /*seed=*/1);
    queries_ = make_vectors(kNQ, kDim, /*seed=*/999);
    labels_.resize(kN);
    for (uint64_t i = 0; i < kN; ++i) {
      labels_[i] = 5000 + i;  // offset labels exercise the id map
    }
    gt_ = ground_truth(data_, kN, kDim, queries_, kNQ, kTopK);

    auto base = std::filesystem::temp_directory_path();
    pq_dir_ = (base / "diskann_e2e_pq").string();
    nopq_dir_ = (base / "diskann_e2e_nopq").string();
    std::error_code ec;
    std::filesystem::remove_all(pq_dir_, ec);
    std::filesystem::remove_all(nopq_dir_, ec);

    DiskANNBuildParams bp;
    bp.R = 64;
    bp.L = 125;
    bp.alpha = 1.2f;
    bp.cache_ratio = 0.05;
    bp.pq_n_chunks = 32;  // 128 / 32 = chunk_dim 4
    DiskANNIndex::build(pq_dir_, data_.data(), labels_.data(), kN, kDim, bp);

    bp.pq_n_chunks = 0;  // no-PQ variant on the same data
    DiskANNIndex::build(nopq_dir_, data_.data(), labels_.data(), kN, kDim, bp);
  }

  static void TearDownTestSuite() {
    std::error_code ec;
    std::filesystem::remove_all(pq_dir_, ec);
    std::filesystem::remove_all(nopq_dir_, ec);
    data_.clear();
    queries_.clear();
    labels_.clear();
    gt_.clear();
  }

  // Recall@k of a returned label set vs ground-truth internal ids.
  double recall_at_k(const std::vector<uint64_t> &out_labels, uint32_t q, uint32_t k) const {
    std::unordered_set<uint32_t> truth(gt_[q].begin(), gt_[q].end());
    uint32_t hit = 0;
    for (uint32_t i = 0; i < k; ++i) {
      if (out_labels[i] == DiskANNIndex::kNoLabel) {
        continue;
      }
      const uint32_t internal = static_cast<uint32_t>(out_labels[i] - 5000);
      if (truth.count(internal) != 0) {
        ++hit;
      }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
  }

  double mean_recall(DiskANNIndex &idx, const DiskANNSearchParams &sp, uint32_t nq = kNQ) const {
    double sum = 0.0;
    std::vector<uint64_t> out_l(kTopK);
    std::vector<float> out_d(kTopK);
    for (uint32_t q = 0; q < nq; ++q) {
      idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK, out_l.data(),
                 out_d.data(), sp);
      sum += recall_at_k(out_l, q, kTopK);
    }
    return sum / nq;
  }

  static std::vector<float> data_;
  static std::vector<float> queries_;
  static std::vector<uint64_t> labels_;
  static std::vector<std::vector<uint32_t>> gt_;
  static std::string pq_dir_;
  static std::string nopq_dir_;
};

std::vector<float> DiskANNE2ETest::data_;
std::vector<float> DiskANNE2ETest::queries_;
std::vector<uint64_t> DiskANNE2ETest::labels_;
std::vector<std::vector<uint32_t>> DiskANNE2ETest::gt_;
std::string DiskANNE2ETest::pq_dir_;
std::string DiskANNE2ETest::nopq_dir_;

TEST_F(DiskANNE2ETest, RecallAbovePoint9_PQRerank) {
  DiskANNIndex idx;
  idx.load(pq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const DiskANNSearchParams sp{/*L=*/150, /*use_pq=*/true, /*rerank=*/true, /*rerank_count=*/100};
  const double recall = mean_recall(idx, sp);
  std::cout << "[e2e] PQ+rerank recall@10 = " << recall << std::endl;
  EXPECT_GT(recall, 0.9);
}

TEST_F(DiskANNE2ETest, NoPQReturnsCorrectResults) {
  // No-PQ is exact disk greedy search, so recall is high at a moderate L. It
  // reads every discovered node (higher I/O than PQ, design D5), so this uses a
  // smaller query subset to keep the test fast.
  DiskANNIndex idx;
  idx.load(nopq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const DiskANNSearchParams sp{/*L=*/150, /*use_pq=*/false, /*rerank=*/false};
  const double recall = mean_recall(idx, sp, /*nq=*/25);
  std::cout << "[e2e] No-PQ (exact greedy) recall@10 = " << recall << std::endl;
  EXPECT_GT(recall, 0.9);
}

TEST_F(DiskANNE2ETest, NoPQAsyncMatchesDeterministicRecall) {
  // No-PQ distances are exact and the frontier insert is order-independent, so
  // the async-pipelined default and the deterministic barrier reach the same
  // recall (they may differ only in tie-ordering, to which recall@k is blind;
  // the relaxed expansion order can perturb the visited set marginally, hence a
  // small tolerance rather than exact equality).
  DiskANNIndex idx;
  idx.load(nopq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const double r_async = mean_recall(
      idx, {/*L=*/150, /*use_pq=*/false, /*rerank=*/false, /*rerank_count=*/0,
            /*deterministic=*/false},
      /*nq=*/25);
  const double r_det = mean_recall(
      idx, {/*L=*/150, /*use_pq=*/false, /*rerank=*/false, /*rerank_count=*/0,
            /*deterministic=*/true},
      /*nq=*/25);
  std::cout << "[e2e] No-PQ recall async=" << r_async << " deterministic=" << r_det << std::endl;
  EXPECT_NEAR(r_async, r_det, 0.02);
}

TEST_F(DiskANNE2ETest, RerankImprovesRecall) {
  DiskANNIndex idx;
  idx.load(pq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const double r_no = mean_recall(idx, {/*L=*/150, /*use_pq=*/true, /*rerank=*/false});
  const double r_yes = mean_recall(idx, {/*L=*/150, /*use_pq=*/true, /*rerank=*/true});
  std::cout << "[e2e] recall no-rerank=" << r_no << " rerank=" << r_yes << std::endl;
  EXPECT_GE(r_yes, r_no);
}

TEST_F(DiskANNE2ETest, BatchMatchesSequential) {
  DiskANNIndex idx;
  idx.load(pq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  // deterministic=true: this test asserts byte-exact batch/sequential equality,
  // which is exactly the guarantee the deterministic per-beam barrier provides.
  const DiskANNSearchParams sp{/*L=*/120, /*use_pq=*/true, /*rerank=*/true,
                               /*rerank_count=*/0, /*deterministic=*/true};

  std::vector<uint64_t> seq_l(static_cast<uint64_t>(kNQ) * kTopK);
  std::vector<float> seq_d(static_cast<uint64_t>(kNQ) * kTopK);
  for (uint32_t q = 0; q < kNQ; ++q) {
    idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK,
               seq_l.data() + static_cast<uint64_t>(q) * kTopK,
               seq_d.data() + static_cast<uint64_t>(q) * kTopK, sp);
  }

  std::vector<uint64_t> bat_l(static_cast<uint64_t>(kNQ) * kTopK);
  std::vector<float> bat_d(static_cast<uint64_t>(kNQ) * kTopK);
  idx.batch_search(queries_.data(), kNQ, kTopK, bat_l.data(), bat_d.data(), /*num_threads=*/4, sp);

  EXPECT_EQ(seq_l, bat_l);
  for (size_t i = 0; i < seq_d.size(); ++i) {
    EXPECT_FLOAT_EQ(seq_d[i], bat_d[i]) << "i=" << i;
  }
}

TEST_F(DiskANNE2ETest, ConcurrentSearchCorrect) {
  DiskANNIndex idx;
  idx.load(pq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  // deterministic=true: this test asserts byte-exact batch/sequential equality,
  // which is exactly the guarantee the deterministic per-beam barrier provides.
  const DiskANNSearchParams sp{/*L=*/120, /*use_pq=*/true, /*rerank=*/true,
                               /*rerank_count=*/0, /*deterministic=*/true};

  // Sequential reference.
  std::vector<uint64_t> ref(static_cast<uint64_t>(kNQ) * kTopK);
  for (uint32_t q = 0; q < kNQ; ++q) {
    std::vector<float> d(kTopK);
    idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK,
               ref.data() + static_cast<uint64_t>(q) * kTopK, d.data(), sp);
  }

  // 4 threads each run a disjoint slice concurrently.
  std::vector<uint64_t> conc(static_cast<uint64_t>(kNQ) * kTopK);
  auto worker = [&](uint32_t begin, uint32_t end) {
    std::vector<float> d(kTopK);
    for (uint32_t q = begin; q < end; ++q) {
      idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK,
                 conc.data() + static_cast<uint64_t>(q) * kTopK, d.data(), sp);
    }
  };
  std::vector<std::thread> threads;
  const uint32_t chunk = (kNQ + 3) / 4;
  for (uint32_t t = 0; t < 4; ++t) {
    const uint32_t b = t * chunk;
    const uint32_t e = std::min(kNQ, b + chunk);
    if (b < e) {
      threads.emplace_back(worker, b, e);
    }
  }
  for (auto &th : threads) {
    th.join();
  }
  EXPECT_EQ(ref, conc);
}

TEST_F(DiskANNE2ETest, ConcurrentDefaultPathValid) {
  // The default (deterministic=false) async-pipelined path is non-deterministic
  // in tie-ordering, so byte-equality cannot be asserted. Instead run every query
  // concurrently and require recall to stay high: a data race in the async
  // pipeline (free_slots / inflight / scratch reuse) would corrupt results and
  // tank recall.
  DiskANNIndex idx;
  idx.load(pq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const DiskANNSearchParams sp{/*L=*/150, /*use_pq=*/true, /*rerank=*/true,
                               /*rerank_count=*/100, /*deterministic=*/false};

  std::vector<uint64_t> out(static_cast<uint64_t>(kNQ) * kTopK);
  auto worker = [&](uint32_t begin, uint32_t end) {
    std::vector<float> d(kTopK);
    for (uint32_t q = begin; q < end; ++q) {
      idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK,
                 out.data() + static_cast<uint64_t>(q) * kTopK, d.data(), sp);
    }
  };
  std::vector<std::thread> threads;
  const uint32_t chunk = (kNQ + 3) / 4;
  for (uint32_t t = 0; t < 4; ++t) {
    const uint32_t b = t * chunk;
    const uint32_t e = std::min(kNQ, b + chunk);
    if (b < e) {
      threads.emplace_back(worker, b, e);
    }
  }
  for (auto &th : threads) {
    th.join();
  }

  double sum = 0.0;
  std::vector<uint64_t> row(kTopK);
  for (uint32_t q = 0; q < kNQ; ++q) {
    for (uint32_t i = 0; i < kTopK; ++i) {
      row[i] = out[static_cast<uint64_t>(q) * kTopK + i];
    }
    sum += recall_at_k(row, q, kTopK);
  }
  const double recall = sum / kNQ;
  std::cout << "[e2e] concurrent default-path recall@10 = " << recall << std::endl;
  EXPECT_GT(recall, 0.9);
}

TEST_F(DiskANNE2ETest, NoPQConcurrentDefaultPathValid) {
  // Same race check for the No-PQ async pipeline, which adds a `pending` deque on
  // top of the free_slots/inflight machinery. Run all queries concurrently on the
  // default path and require high recall (a data race would corrupt and tank it).
  DiskANNIndex idx;
  idx.load(nopq_dir_, {/*num_threads=*/4, /*beam_width=*/8});
  const DiskANNSearchParams sp{/*L=*/150, /*use_pq=*/false, /*rerank=*/false,
                               /*rerank_count=*/0, /*deterministic=*/false};

  std::vector<uint64_t> out(static_cast<uint64_t>(kNQ) * kTopK);
  auto worker = [&](uint32_t begin, uint32_t end) {
    std::vector<float> d(kTopK);
    for (uint32_t q = begin; q < end; ++q) {
      idx.search(queries_.data() + static_cast<uint64_t>(q) * kDim, kTopK,
                 out.data() + static_cast<uint64_t>(q) * kTopK, d.data(), sp);
    }
  };
  std::vector<std::thread> threads;
  const uint32_t chunk = (kNQ + 3) / 4;
  for (uint32_t t = 0; t < 4; ++t) {
    const uint32_t b = t * chunk;
    const uint32_t e = std::min(kNQ, b + chunk);
    if (b < e) {
      threads.emplace_back(worker, b, e);
    }
  }
  for (auto &th : threads) {
    th.join();
  }

  double sum = 0.0;
  std::vector<uint64_t> row(kTopK);
  for (uint32_t q = 0; q < kNQ; ++q) {
    for (uint32_t i = 0; i < kTopK; ++i) {
      row[i] = out[static_cast<uint64_t>(q) * kTopK + i];
    }
    sum += recall_at_k(row, q, kTopK);
  }
  const double recall = sum / kNQ;
  std::cout << "[e2e] No-PQ concurrent default-path recall@10 = " << recall << std::endl;
  EXPECT_GT(recall, 0.9);
}

}  // namespace
