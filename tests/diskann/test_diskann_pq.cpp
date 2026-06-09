// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/pq_table.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

#include "simd/distance_l2.hpp"

namespace {

using alaya::diskann::kPQNumCentroids;
using alaya::diskann::PQTable;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 11) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

float exact_l2_sqr(const float *a, const float *b, uint64_t dim) {
  return alaya::simd::l2_sqr<float, float>(a, b, dim);
}

class PQTableTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto &p : owned_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  }
  std::filesystem::path temp_path(const std::string &tag) {
    static std::atomic<uint64_t> counter{0};
    auto p = std::filesystem::temp_directory_path() /
             ("diskann_pq_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".bin");
    owned_.push_back(p);
    return p;
  }
  std::vector<std::filesystem::path> owned_;
};

// --- train shape / validation -----------------------------------------------

TEST_F(PQTableTest, TrainRejectsIndivisibleDim) {
  const auto data = make_vectors(10, 100);
  PQTable pq;
  EXPECT_THROW(pq.train(data.data(), 10, /*dim=*/100, /*n_chunks=*/64), std::invalid_argument);
}

TEST_F(PQTableTest, TrainBalancedChunks) {
  const uint64_t n = 300, dim = 128;
  const uint32_t n_chunks = 64;
  const auto data = make_vectors(n, dim);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  EXPECT_EQ(pq.dim(), dim);
  EXPECT_EQ(pq.n_chunks(), n_chunks);
  EXPECT_EQ(pq.chunk_dim(), 2u);  // 128 / 64
  EXPECT_EQ(pq.codebook().size(),
            static_cast<size_t>(n_chunks) * kPQNumCentroids * 2);  // n_chunks*256*chunk_dim
  EXPECT_EQ(pq.global_centroid().size(), dim);
}

TEST_F(PQTableTest, EncodeShape) {
  const uint64_t n = 1000, dim = 64;
  const uint32_t n_chunks = 32;
  const auto data = make_vectors(n, dim);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);
  EXPECT_EQ(pq.codes().size(), static_cast<size_t>(n) * n_chunks);
  EXPECT_EQ(pq.num_points(), n);
}

// --- encode correctness (deterministic small example) -----------------------

TEST_F(PQTableTest, EncodeSmallExampleIsDeterministic) {
  // With n <= 256, centroids == training residuals, so each point encodes to
  // its own unique index in every chunk.
  const uint64_t n = 10, dim = 8;
  const uint32_t n_chunks = 2;
  const auto data = make_vectors(n, dim, /*seed=*/42);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);
  const auto &codes = pq.codes();
  for (uint64_t i = 0; i < n; ++i) {
    for (uint32_t c = 0; c < n_chunks; ++c) {
      EXPECT_EQ(codes[i * n_chunks + c], static_cast<uint8_t>(i)) << "i=" << i << " c=" << c;
    }
  }
}

// --- PQ distance equals true L2 in the lossless (n <= 256) regime ------------

TEST_F(PQTableTest, PQDistanceEqualsTrueL2WhenLossless) {
  const uint64_t n = 50, dim = 16;
  const uint32_t n_chunks = 4;
  const auto data = make_vectors(n, dim, /*seed=*/7);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);

  std::vector<float> table(static_cast<size_t>(n_chunks) * kPQNumCentroids);
  for (uint64_t qi = 0; qi < n; ++qi) {
    pq.preprocess_query(data.data() + qi * dim, table.data());
    for (uint64_t pi = 0; pi < n; ++pi) {
      const float approx = pq.pq_distance(pi, table.data());
      const float truth = exact_l2_sqr(data.data() + qi * dim, data.data() + pi * dim, dim);
      EXPECT_NEAR(approx, truth, 1e-2f * (1.0f + truth)) << "qi=" << qi << " pi=" << pi;
    }
  }
}

// --- distance table values against a hand-built codebook --------------------

TEST_F(PQTableTest, PreprocessQueryMatchesManualTable) {
  const uint64_t dim = 4;
  const uint32_t n_chunks = 2, chunk_dim = 2;
  std::vector<float> gc(dim, 0.0f);  // zero global centroid => residual == query
  std::vector<float> codebook(static_cast<size_t>(n_chunks) * kPQNumCentroids * chunk_dim, 0.0f);
  // centroid (c, k) = (k, k)
  for (uint32_t c = 0; c < n_chunks; ++c) {
    for (uint32_t k = 0; k < kPQNumCentroids; ++k) {
      float *cent = codebook.data() + (static_cast<size_t>(c) * kPQNumCentroids + k) * chunk_dim;
      cent[0] = static_cast<float>(k);
      cent[1] = static_cast<float>(k);
    }
  }
  PQTable pq = PQTable::from_codebook(dim, n_chunks, gc, codebook);

  const std::vector<float> query = {3.0f, 3.0f, 5.0f, 5.0f};  // chunk0=(3,3) chunk1=(5,5)
  std::vector<float> table(static_cast<size_t>(n_chunks) * kPQNumCentroids);
  pq.preprocess_query(query.data(), table.data());

  for (uint32_t k = 0; k < kPQNumCentroids; ++k) {
    const float e0 = 2.0f * (3.0f - k) * (3.0f - k);
    const float e1 = 2.0f * (5.0f - k) * (5.0f - k);
    EXPECT_NEAR(table[0 * kPQNumCentroids + k], e0, 1e-3f) << "chunk0 k=" << k;
    EXPECT_NEAR(table[1 * kPQNumCentroids + k], e1, 1e-3f) << "chunk1 k=" << k;
  }
}

TEST_F(PQTableTest, PQDistanceIsSumOfTableLookups) {
  const uint64_t n = 400, dim = 32;  // n > 256 exercises real k-means
  const uint32_t n_chunks = 8;
  const auto data = make_vectors(n, dim);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);

  std::vector<float> table(static_cast<size_t>(n_chunks) * kPQNumCentroids);
  pq.preprocess_query(data.data() + 5 * dim, table.data());
  const auto &codes = pq.codes();
  for (uint64_t pi : {0ull, 1ull, 100ull, 399ull}) {
    float expected = 0.0f;
    for (uint32_t c = 0; c < n_chunks; ++c) {
      expected += table[static_cast<size_t>(c) * kPQNumCentroids + codes[pi * n_chunks + c]];
    }
    EXPECT_FLOAT_EQ(pq.pq_distance(pi, table.data()), expected) << "pi=" << pi;
  }
}

// --- parallel train/encode is byte-identical to serial ----------------------

TEST_F(PQTableTest, ParallelTrainEncodeByteIdenticalToSerial) {
  // Chunks train independently (own seed seed+c, disjoint codebook region) and
  // each point encodes independently, so multi-threaded train()/encode() must
  // produce a byte-identical codebook and codes to the single-threaded run. (Each
  // chunk's k-means stays serial internally, so there is no FP-reduction reorder.)
  const uint64_t n = 2000, dim = 64;  // n > 256 => real k-means
  const uint32_t n_chunks = 16;
  const auto data = make_vectors(n, dim, /*seed=*/123);

  PQTable serial;
  serial.train(data.data(), n, dim, n_chunks, /*n_iters=*/15, /*seed=*/1234, /*num_threads=*/1);
  serial.encode(data.data(), n, /*num_threads=*/1);

  PQTable parallel;
  parallel.train(data.data(), n, dim, n_chunks, /*n_iters=*/15, /*seed=*/1234, /*num_threads=*/8);
  parallel.encode(data.data(), n, /*num_threads=*/8);

  ASSERT_EQ(serial.codebook().size(), parallel.codebook().size());
  EXPECT_EQ(serial.codebook(), parallel.codebook());  // bit-identical float codebook
  ASSERT_EQ(serial.codes().size(), parallel.codes().size());
  EXPECT_EQ(serial.codes(), parallel.codes());
  EXPECT_EQ(serial.global_centroid(), parallel.global_centroid());
}

// --- persistence ------------------------------------------------------------

TEST_F(PQTableTest, FileRoundtripBitIdentical) {
  const uint64_t n = 500, dim = 48;
  const uint32_t n_chunks = 12;
  const auto data = make_vectors(n, dim);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);

  const auto pivots = temp_path("pivots");
  const auto compressed = temp_path("compressed");
  pq.save(pivots.string(), compressed.string());

  PQTable loaded;
  loaded.load(pivots.string(), compressed.string(), n, dim, n_chunks);

  ASSERT_EQ(loaded.global_centroid().size(), pq.global_centroid().size());
  for (size_t i = 0; i < pq.global_centroid().size(); ++i) {
    EXPECT_EQ(loaded.global_centroid()[i], pq.global_centroid()[i]) << "gc i=" << i;
  }
  ASSERT_EQ(loaded.codebook().size(), pq.codebook().size());
  for (size_t i = 0; i < pq.codebook().size(); ++i) {
    EXPECT_EQ(loaded.codebook()[i], pq.codebook()[i]) << "cb i=" << i;
  }
  ASSERT_EQ(loaded.codes().size(), pq.codes().size());
  EXPECT_EQ(loaded.codes(), pq.codes());

  // And the loaded table produces identical distances.
  std::vector<float> t0(static_cast<size_t>(n_chunks) * kPQNumCentroids);
  std::vector<float> t1(static_cast<size_t>(n_chunks) * kPQNumCentroids);
  pq.preprocess_query(data.data(), t0.data());
  loaded.preprocess_query(data.data(), t1.data());
  for (uint64_t pi : {0ull, 250ull, 499ull}) {
    EXPECT_FLOAT_EQ(pq.pq_distance(pi, t0.data()), loaded.pq_distance(pi, t1.data())) << "pi=" << pi;
  }
}

TEST_F(PQTableTest, LoadRejectsWrongSize) {
  const uint64_t n = 100, dim = 16;
  const uint32_t n_chunks = 4;
  const auto data = make_vectors(n, dim);
  PQTable pq;
  pq.train(data.data(), n, dim, n_chunks);
  pq.encode(data.data(), n);
  const auto pivots = temp_path("pivots_bad");
  const auto compressed = temp_path("compressed_bad");
  pq.save(pivots.string(), compressed.string());

  PQTable loaded;
  // Wrong n => pq_compressed.bin size check fails.
  EXPECT_THROW(loaded.load(pivots.string(), compressed.string(), n + 1, dim, n_chunks),
               std::runtime_error);
  // Wrong dim => pq_pivots.bin size check fails.
  EXPECT_THROW(loaded.load(pivots.string(), compressed.string(), n, dim * 2, n_chunks),
               std::runtime_error);
}

TEST_F(PQTableTest, FromCodebookRejectsBadShapes) {
  EXPECT_THROW(PQTable::from_codebook(4, 3, std::vector<float>(4), std::vector<float>(4)),
               std::invalid_argument);  // 4 % 3 != 0
  EXPECT_THROW(
      PQTable::from_codebook(4, 2, std::vector<float>(3), std::vector<float>(2 * 256 * 2)),
      std::invalid_argument);  // global centroid wrong size
  EXPECT_THROW(PQTable::from_codebook(4, 2, std::vector<float>(4), std::vector<float>(10)),
               std::invalid_argument);  // codebook wrong size
}

}  // namespace
