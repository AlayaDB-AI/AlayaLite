// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file pq_table.hpp
 * @brief Product Quantization (PQ) table: train / encode / per-query distance.
 *
 * PQ splits the @c dim -dimensional space into @c n_chunks equal sub-spaces and
 * learns a 256-entry codebook per sub-space via k-means. Each vector becomes
 * @c n_chunks uint8 codes. At search time, the query is turned once into an
 * @c n_chunks x 256 distance table, and the approximate distance to any stored
 * point is a sum of @c n_chunks table lookups (no per-neighbor arithmetic).
 *
 * Encoding is done on centroid-residual vectors (x - global_centroid). Because
 * the global centroid cancels in the difference, the PQ distance approximates
 * the true squared L2 distance between query and point.
 *
 * The trained table (global centroid + codebook + codes) is immutable after
 * load and may be shared read-only across search threads; the per-query
 * distance table is owned by each thread's scratch (see search_scratch.hpp).
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "simd/distance_l2.hpp"

namespace alaya::diskann {

/// Number of centroids per chunk (uint8 codes => 256).
inline constexpr uint32_t kPQNumCentroids = 256;

class PQTable {
 public:
  PQTable() = default;

  // --- Build-time ----------------------------------------------------------

  /**
   * @brief Train per-chunk k-means codebooks on @p n row-major vectors.
   *
   * @param data     Row-major @c n*dim float32 training vectors.
   * @param n        Number of training vectors (> 0).
   * @param dim      Vector dimension (> 0, divisible by @p n_chunks).
   * @param n_chunks Number of PQ sub-spaces (> 0).
   * @param n_iters  Lloyd iterations for k-means (used when n > 256).
   * @param seed     RNG seed for reproducible k-means++ initialization.
   * @param num_threads Workers for the per-chunk training (0 => all cores).
   *         Chunks are independent, so the result is byte-identical to the
   *         single-threaded run regardless of thread count.
   *
   * @throws std::invalid_argument if dim is not divisible by n_chunks, or any
   *         size is zero.
   */
  void train(const float *data,
             uint64_t n,
             uint64_t dim,
             uint32_t n_chunks,
             uint32_t n_iters = 15,
             uint64_t seed = 1234,
             uint32_t num_threads = 0) {
    if (data == nullptr) {
      throw std::invalid_argument("PQTable::train: data is null");
    }
    if (n == 0 || dim == 0 || n_chunks == 0) {
      throw std::invalid_argument("PQTable::train: n/dim/n_chunks must be > 0");
    }
    if (dim % n_chunks != 0) {
      throw std::invalid_argument("PQTable::train: dim (" + std::to_string(dim) +
                                  ") must be divisible by n_chunks (" + std::to_string(n_chunks) +
                                  ")");
    }
    dim_ = dim;
    n_chunks_ = n_chunks;
    chunk_dim_ = static_cast<uint32_t>(dim / n_chunks);
    ensure_l2();

    // Global centroid = mean of all training vectors.
    global_centroid_.assign(dim_, 0.0f);
    for (uint64_t i = 0; i < n; ++i) {
      const float *v = data + i * dim_;
      for (uint64_t d = 0; d < dim_; ++d) {
        global_centroid_[d] += v[d];
      }
    }
    for (uint64_t d = 0; d < dim_; ++d) {
      global_centroid_[d] /= static_cast<float>(n);
    }

    // Residuals (n * dim), contiguous so each chunk is a contiguous stride.
    std::vector<float> residual(static_cast<size_t>(n) * dim_);
    for (uint64_t i = 0; i < n; ++i) {
      const float *v = data + i * dim_;
      float *r = residual.data() + i * dim_;
      for (uint64_t d = 0; d < dim_; ++d) {
        r[d] = v[d] - global_centroid_[d];
      }
    }

    codebook_.assign(static_cast<size_t>(n_chunks_) * kPQNumCentroids * chunk_dim_, 0.0f);

    // Per-chunk training. Chunks are independent — each writes a disjoint codebook
    // region and trains with its own seed (seed + c) — so they parallelize with a
    // byte-identical result regardless of thread count (completion order is
    // irrelevant). Each worker owns its chunk_data buffer and pulls chunks off a
    // shared counter.
    const uint32_t hw = std::max<uint32_t>(1, std::thread::hardware_concurrency());
    const uint32_t workers = std::min<uint32_t>(num_threads == 0 ? hw : num_threads, n_chunks_);
    std::atomic<uint32_t> next_chunk{0};
    auto chunk_worker = [&]() {
      std::vector<float> chunk_data(static_cast<size_t>(n) * chunk_dim_);
      for (;;) {
        const uint32_t c = next_chunk.fetch_add(1, std::memory_order_relaxed);
        if (c >= n_chunks_) {
          break;
        }
        for (uint64_t i = 0; i < n; ++i) {
          std::memcpy(chunk_data.data() + i * chunk_dim_,
                      residual.data() + i * dim_ + static_cast<uint64_t>(c) * chunk_dim_,
                      chunk_dim_ * sizeof(float));
        }
        float *centroids = codebook_.data() + static_cast<size_t>(c) * kPQNumCentroids * chunk_dim_;
        train_chunk(chunk_data.data(), n, centroids, n_iters, seed + c);
      }
    };
    if (workers <= 1) {
      chunk_worker();
    } else {
      std::vector<std::thread> pool;
      pool.reserve(workers);
      for (uint32_t t = 0; t < workers; ++t) {
        pool.emplace_back(chunk_worker);
      }
      for (auto &th : pool) {
        th.join();
      }
    }
  }

  /**
   * @brief Encode @p n vectors into @c n*n_chunks uint8 codes (row-major).
   *
   * Stores the codes internally (queryable via pq_distance / codes()). Each
   * vector's code row is independent, so encoding parallelizes across points
   * (@p num_threads, 0 => all cores) with a byte-identical result.
   * @throws std::logic_error if called before train()/load().
   */
  void encode(const float *data, uint64_t n, uint32_t num_threads = 0) {
    if (codebook_.empty()) {
      throw std::logic_error("PQTable::encode: not trained");
    }
    if (data == nullptr || n == 0) {
      throw std::invalid_argument("PQTable::encode: data null or n == 0");
    }
    ensure_l2();
    num_points_ = n;
    codes_.assign(static_cast<size_t>(n) * n_chunks_, 0);

    const uint32_t hw = std::max<uint32_t>(1, std::thread::hardware_concurrency());
    const uint32_t want = num_threads == 0 ? hw : num_threads;
    const uint32_t workers =
        static_cast<uint32_t>(std::min<uint64_t>(want, std::max<uint64_t>(1, n)));
    std::atomic<uint64_t> next_block{0};
    constexpr uint64_t kBlock = 4096;
    auto encode_worker = [&]() {
      std::vector<float> r(dim_);
      for (;;) {
        const uint64_t start = next_block.fetch_add(kBlock, std::memory_order_relaxed);
        if (start >= n) {
          break;
        }
        const uint64_t end = std::min<uint64_t>(start + kBlock, n);
        for (uint64_t i = start; i < end; ++i) {
          const float *v = data + i * dim_;
          for (uint64_t d = 0; d < dim_; ++d) {
            r[d] = v[d] - global_centroid_[d];
          }
          uint8_t *code_row = codes_.data() + i * n_chunks_;
          for (uint32_t c = 0; c < n_chunks_; ++c) {
            code_row[c] = nearest_centroid(r.data() + static_cast<uint64_t>(c) * chunk_dim_, c);
          }
        }
      }
    };
    if (workers <= 1) {
      encode_worker();
    } else {
      std::vector<std::thread> pool;
      pool.reserve(workers);
      for (uint32_t t = 0; t < workers; ++t) {
        pool.emplace_back(encode_worker);
      }
      for (auto &th : pool) {
        th.join();
      }
    }
  }

  // --- Search-time ---------------------------------------------------------

  /**
   * @brief Precompute the @c n_chunks x 256 query distance table.
   *
   * @param query      Query vector (dim float32).
   * @param table_out  Caller-owned buffer of @c n_chunks*256 float32. Entry
   *                   [c*256 + k] = squared L2 between the query's chunk-c
   *                   residual and centroid k of chunk c.
   */
  void preprocess_query(const float *query, float *table_out) const {
    std::vector<float> qres(dim_);
    for (uint64_t d = 0; d < dim_; ++d) {
      qres[d] = query[d] - global_centroid_[d];
    }
    for (uint32_t c = 0; c < n_chunks_; ++c) {
      const float *qchunk = qres.data() + static_cast<uint64_t>(c) * chunk_dim_;
      const float *cent = codebook_.data() + static_cast<size_t>(c) * kPQNumCentroids * chunk_dim_;
      float *trow = table_out + static_cast<size_t>(c) * kPQNumCentroids;
      for (uint32_t k = 0; k < kPQNumCentroids; ++k) {
        trow[k] = l2_(qchunk, cent + static_cast<size_t>(k) * chunk_dim_, chunk_dim_);
      }
    }
  }

  /**
   * @brief Approximate distance from query to point @p point_id.
   * @param point_id   Index into the stored code array.
   * @param dist_table Table produced by preprocess_query().
   * @return sum over chunks of dist_table[c*256 + code(point_id, c)].
   */
  [[nodiscard]] float pq_distance(uint64_t point_id, const float *dist_table) const {
    const uint8_t *code_row = codes_.data() + point_id * n_chunks_;
    float sum = 0.0f;
    for (uint32_t c = 0; c < n_chunks_; ++c) {
      sum += dist_table[static_cast<size_t>(c) * kPQNumCentroids + code_row[c]];
    }
    return sum;
  }

  // --- Persistence ---------------------------------------------------------

  /**
   * @brief Write pq_pivots.bin (global centroid + codebook) and
   *        pq_compressed.bin (codes). Pure payload, no header (shape lives in
   *        the index meta.bin).
   */
  void save(const std::string &pivots_path, const std::string &compressed_path) const {
    if (codebook_.empty()) {
      throw std::logic_error("PQTable::save: not trained");
    }
    {
      std::ofstream out(pivots_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("PQTable::save: cannot open " + pivots_path);
      }
      out.write(reinterpret_cast<const char *>(global_centroid_.data()),
                static_cast<std::streamsize>(global_centroid_.size() * sizeof(float)));
      out.write(reinterpret_cast<const char *>(codebook_.data()),
                static_cast<std::streamsize>(codebook_.size() * sizeof(float)));
      if (!out) {
        throw std::runtime_error("PQTable::save: write failed for " + pivots_path);
      }
    }
    {
      std::ofstream out(compressed_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("PQTable::save: cannot open " + compressed_path);
      }
      out.write(reinterpret_cast<const char *>(codes_.data()),
                static_cast<std::streamsize>(codes_.size()));
      if (!out) {
        throw std::runtime_error("PQTable::save: write failed for " + compressed_path);
      }
    }
  }

  /**
   * @brief Load pq_pivots.bin + pq_compressed.bin. Shape (n, dim, n_chunks) is
   *        supplied by the caller (from meta.bin) and validated against file
   *        sizes.
   */
  void load(const std::string &pivots_path,
            const std::string &compressed_path,
            uint64_t n,
            uint64_t dim,
            uint32_t n_chunks) {
    if (n == 0 || dim == 0 || n_chunks == 0 || dim % n_chunks != 0) {
      throw std::invalid_argument("PQTable::load: invalid shape");
    }
    dim_ = dim;
    n_chunks_ = n_chunks;
    chunk_dim_ = static_cast<uint32_t>(dim / n_chunks);
    num_points_ = n;
    ensure_l2();

    const size_t centroid_floats = dim_;
    const size_t codebook_floats = static_cast<size_t>(n_chunks_) * kPQNumCentroids * chunk_dim_;
    const uint64_t expect_pivots = (centroid_floats + codebook_floats) * sizeof(float);

    std::error_code ec;
    const auto pivots_size = std::filesystem::file_size(pivots_path, ec);
    if (ec) {
      throw std::runtime_error("PQTable::load: cannot stat " + pivots_path);
    }
    if (static_cast<uint64_t>(pivots_size) != expect_pivots) {
      throw std::runtime_error("PQTable::load: " + pivots_path + " size " +
                               std::to_string(pivots_size) + " != expected " +
                               std::to_string(expect_pivots));
    }
    std::ifstream pin(pivots_path, std::ios::binary);
    if (!pin) {
      throw std::runtime_error("PQTable::load: cannot open " + pivots_path);
    }
    global_centroid_.assign(centroid_floats, 0.0f);
    codebook_.assign(codebook_floats, 0.0f);
    pin.read(reinterpret_cast<char *>(global_centroid_.data()),
             static_cast<std::streamsize>(centroid_floats * sizeof(float)));
    pin.read(reinterpret_cast<char *>(codebook_.data()),
             static_cast<std::streamsize>(codebook_floats * sizeof(float)));
    if (!pin) {
      throw std::runtime_error("PQTable::load: short read on " + pivots_path);
    }

    const uint64_t expect_codes = n * n_chunks_;
    const auto codes_size = std::filesystem::file_size(compressed_path, ec);
    if (ec) {
      throw std::runtime_error("PQTable::load: cannot stat " + compressed_path);
    }
    if (static_cast<uint64_t>(codes_size) != expect_codes) {
      throw std::runtime_error("PQTable::load: " + compressed_path + " size " +
                               std::to_string(codes_size) + " != expected " +
                               std::to_string(expect_codes));
    }
    std::ifstream cin(compressed_path, std::ios::binary);
    if (!cin) {
      throw std::runtime_error("PQTable::load: cannot open " + compressed_path);
    }
    codes_.assign(expect_codes, 0);
    cin.read(reinterpret_cast<char *>(codes_.data()), static_cast<std::streamsize>(expect_codes));
    if (!cin) {
      throw std::runtime_error("PQTable::load: short read on " + compressed_path);
    }
  }

  /// Build a table directly from arrays (used by tests and external codebooks).
  static PQTable from_codebook(uint64_t dim,
                               uint32_t n_chunks,
                               std::vector<float> global_centroid,
                               std::vector<float> codebook) {
    if (dim == 0 || n_chunks == 0 || dim % n_chunks != 0) {
      throw std::invalid_argument("PQTable::from_codebook: invalid shape");
    }
    PQTable t;
    t.dim_ = dim;
    t.n_chunks_ = n_chunks;
    t.chunk_dim_ = static_cast<uint32_t>(dim / n_chunks);
    if (global_centroid.size() != dim) {
      throw std::invalid_argument("PQTable::from_codebook: global_centroid size");
    }
    if (codebook.size() != static_cast<size_t>(n_chunks) * kPQNumCentroids * t.chunk_dim_) {
      throw std::invalid_argument("PQTable::from_codebook: codebook size");
    }
    t.global_centroid_ = std::move(global_centroid);
    t.codebook_ = std::move(codebook);
    t.ensure_l2();
    return t;
  }

  // --- Accessors -----------------------------------------------------------

  [[nodiscard]] uint64_t num_points() const { return num_points_; }
  [[nodiscard]] uint64_t dim() const { return dim_; }
  [[nodiscard]] uint32_t n_chunks() const { return n_chunks_; }
  [[nodiscard]] uint32_t chunk_dim() const { return chunk_dim_; }
  [[nodiscard]] const std::vector<float> &global_centroid() const { return global_centroid_; }
  [[nodiscard]] const std::vector<float> &codebook() const { return codebook_; }
  [[nodiscard]] const std::vector<uint8_t> &codes() const { return codes_; }

 private:
  void ensure_l2() {
    if (l2_ == nullptr) {
      l2_ = alaya::simd::get_l2_sqr_func();
    }
  }

  /// argmin over the 256 centroids of chunk @p c for a chunk residual vector.
  [[nodiscard]] uint8_t nearest_centroid(const float *chunk_residual, uint32_t c) const {
    const float *cent = codebook_.data() + static_cast<size_t>(c) * kPQNumCentroids * chunk_dim_;
    return static_cast<uint8_t>(argmin_centroid(chunk_residual, cent));
  }

  /**
   * @brief Train one chunk's 256 centroids into @p centroids (256*chunk_dim).
   *
   * For n <= 256 the training points themselves become the centroids (padded),
   * which makes small-input encoding exact and deterministic. For n > 256 we
   * run k-means++ init followed by @p n_iters Lloyd iterations.
   */
  void train_chunk(const float *chunk_data,
                   uint64_t n,
                   float *centroids,
                   uint32_t n_iters,
                   uint64_t seed) const {
    const uint32_t cd = chunk_dim_;
    if (n <= kPQNumCentroids) {
      for (uint64_t i = 0; i < n; ++i) {
        std::memcpy(centroids + i * cd, chunk_data + i * cd, cd * sizeof(float));
      }
      // Pad remaining centroids with a copy of the last real point.
      for (uint64_t k = n; k < kPQNumCentroids; ++k) {
        std::memcpy(centroids + k * cd, chunk_data + (n - 1) * cd, cd * sizeof(float));
      }
      return;
    }

    std::mt19937_64 rng(seed);
    kmeanspp_init(chunk_data, n, centroids, rng);

    std::vector<uint32_t> assign(n, 0);
    std::vector<double> sums(static_cast<size_t>(kPQNumCentroids) * cd, 0.0);
    std::vector<uint64_t> counts(kPQNumCentroids, 0);
    for (uint32_t iter = 0; iter < n_iters; ++iter) {
      // Assignment step.
      for (uint64_t i = 0; i < n; ++i) {
        assign[i] = argmin_centroid(chunk_data + i * cd, centroids);
      }
      // Update step.
      std::fill(sums.begin(), sums.end(), 0.0);
      std::fill(counts.begin(), counts.end(), 0);
      for (uint64_t i = 0; i < n; ++i) {
        const uint32_t a = assign[i];
        counts[a]++;
        const float *p = chunk_data + i * cd;
        double *s = sums.data() + static_cast<size_t>(a) * cd;
        for (uint32_t d = 0; d < cd; ++d) {
          s[d] += p[d];
        }
      }
      for (uint32_t k = 0; k < kPQNumCentroids; ++k) {
        if (counts[k] == 0) {
          // Empty cluster: reseed to a deterministic random training point.
          const uint64_t idx = rng() % n;
          std::memcpy(centroids + static_cast<size_t>(k) * cd,
                      chunk_data + idx * cd,
                      cd * sizeof(float));
          continue;
        }
        const double inv = 1.0 / static_cast<double>(counts[k]);
        const double *s = sums.data() + static_cast<size_t>(k) * cd;
        float *cptr = centroids + static_cast<size_t>(k) * cd;
        for (uint32_t d = 0; d < cd; ++d) {
          cptr[d] = static_cast<float>(s[d] * inv);
        }
      }
    }
  }

  /// argmin over a chunk's 256-centroid block @p centroids (returns the index).
  /// Shared by k-means assignment (train) and encoding.
  [[nodiscard]] uint32_t argmin_centroid(const float *point, const float *centroids) const {
    const uint32_t cd = chunk_dim_;
    float best = std::numeric_limits<float>::max();
    uint32_t best_k = 0;
    for (uint32_t k = 0; k < kPQNumCentroids; ++k) {
      const float d = l2_(point, centroids + static_cast<size_t>(k) * cd, cd);
      if (d < best) {
        best = d;
        best_k = k;
      }
    }
    return best_k;
  }

  /// k-means++ seeding of 256 centroids (D^2 weighting).
  void kmeanspp_init(const float *chunk_data,
                     uint64_t n,
                     float *centroids,
                     std::mt19937_64 &rng) const {
    const uint32_t cd = chunk_dim_;
    std::uniform_int_distribution<uint64_t> pick(0, n - 1);
    const uint64_t first = pick(rng);
    std::memcpy(centroids, chunk_data + first * cd, cd * sizeof(float));

    std::vector<float> d2(n, std::numeric_limits<float>::max());
    for (uint32_t k = 1; k < kPQNumCentroids; ++k) {
      const float *prev = centroids + static_cast<size_t>(k - 1) * cd;
      double total = 0.0;
      for (uint64_t i = 0; i < n; ++i) {
        const float dist = l2_(chunk_data + i * cd, prev, cd);
        if (dist < d2[i]) {
          d2[i] = dist;
        }
        total += d2[i];
      }
      uint64_t chosen = 0;
      if (total <= 0.0) {
        chosen = pick(rng);  // all points coincide with existing centroids
      } else {
        std::uniform_real_distribution<double> u(0.0, total);
        double target = u(rng);
        for (uint64_t i = 0; i < n; ++i) {
          target -= d2[i];
          if (target <= 0.0) {
            chosen = i;
            break;
          }
        }
      }
      std::memcpy(centroids + static_cast<size_t>(k) * cd,
                  chunk_data + chosen * cd,
                  cd * sizeof(float));
    }
  }

  uint64_t dim_ = 0;
  uint32_t n_chunks_ = 0;
  uint32_t chunk_dim_ = 0;
  uint64_t num_points_ = 0;
  std::vector<float> global_centroid_;  // dim
  std::vector<float> codebook_;         // n_chunks * 256 * chunk_dim
  std::vector<uint8_t> codes_;          // num_points * n_chunks
  alaya::simd::L2SqrFunc l2_ = nullptr;
};

}  // namespace alaya::diskann
