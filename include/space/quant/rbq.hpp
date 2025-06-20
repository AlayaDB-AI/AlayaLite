#pragma once

#include "space/rbq_space/data_layout.hpp"
#include "utils/rbq_utils/rbq_impl.hpp"

namespace alaya {
struct EstimateRecord {
  // NOLINTBEGIN
  float ip_x0_qr;
  float est_dist;
  float low_dist;
  // NOLINTEND
  auto operator<(const EstimateRecord &other) const -> bool {
    return this->est_dist < other.est_dist;
  }
};

template <typename DataType, typename IDType>
class SplitSingleQuery {
 private:
  // NOLINTBEGIN
  const DataType *rotated_query_;
  std::vector<uint64_t> QueryBin_;
  DataType G_add_;
  DataType G_k1xSumq_;
  DataType G_kbxSumq_;
  DataType G_error_;
  DataType delta_;
  DataType vl_;
  MetricType metric_type_ = MetricType::L2;
  // NOLINTEND

 public:
  static constexpr size_t kNumBits = 4;
  explicit SplitSingleQuery(const DataType *rotated_query, size_t padded_dim, size_t ex_bits,
                            rabitq_impl::RabitqConfig config,
                            MetricType metric_type = MetricType::L2)
      : rotated_query_(rotated_query), QueryBin_(padded_dim * kNumBits / 64, 0) {
    float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
    float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
    DataType sumq =
        std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<DataType>(0));

    G_k1xSumq_ = sumq * c_1;
    G_kbxSumq_ = sumq * c_b;

    metric_type_ = (metric_type == MetricType::IP) ? MetricType::IP : MetricType::L2;

    std::vector<uint16_t> quant_query = std::vector<uint16_t>(padded_dim);

    // quantize query by rabitq
    std::vector<DataType> centroid(padded_dim, 0);
    rabitq_impl::total_bits::rabitq_scalar_impl<DataType, IDType>(
        rotated_query, centroid.data(), padded_dim, kNumBits, quant_query.data(), delta_, vl_,
        config.t_const, ScalarQuantizerType::RECONSTRUCTION);

    // represent quantized query as u64
    new_transpose_bin(quant_query.data(), QueryBin_.data(), padded_dim, kNumBits);
  }

  [[nodiscard]] auto query_bin() const -> const uint64_t * { return QueryBin_.data(); }

  [[nodiscard]] auto rotated_query() const -> const DataType * { return rotated_query_; }

  [[nodiscard]] auto delta() const -> DataType { return delta_; }

  [[nodiscard]] auto vl() const -> DataType { return vl_; }

  [[nodiscard]] auto k1xsumq() const -> DataType { return G_k1xSumq_; }

  [[nodiscard]] auto kbxsumq() const -> DataType { return G_kbxSumq_; }

  [[nodiscard]] auto g_add() const -> DataType { return G_add_; }

  [[nodiscard]] auto g_error() const -> DataType { return G_error_; }

  void set_g_add(DataType norm, DataType ip = 0) {
    if (metric_type_ == MetricType::L2) {
      G_add_ = norm * norm;
      G_error_ = norm;
    } else if (metric_type_ == MetricType::IP) {
      G_add_ = -ip;
      G_error_ = norm;
    }
  }

  void set_g_error(DataType norm) { G_error_ = norm; }
};

template <typename DataType>
inline void quantize_compact_one_bit(const DataType *data, const DataType *centroid,
                                     size_t padded_dim,
                                     char *bin_data,  // NOLINT
                                     MetricType metric_type = MetricType::L2) {
  BinDataMap<DataType> cur_bin_data(bin_data, padded_dim);

  rabitq_impl::one_bit::one_bit_compact_code(data, centroid, padded_dim, cur_bin_data.bin_code(),
                                             cur_bin_data.f_add(), cur_bin_data.f_rescale(),
                                             cur_bin_data.f_error(), metric_type);
}

template <typename DataType>
inline void quantize_compact_ex_bits(
    const DataType *data, const DataType *centroid, size_t padded_dim, size_t ex_bits,
    char *ex_data,  // NOLINT
    MetricType metric_type = MetricType::L2,
    rabitq_impl::RabitqConfig config = rabitq_impl::RabitqConfig()) {
  ExDataMap<DataType> cur_ex_data(ex_data, padded_dim, ex_bits);

  // we do not use this error factor here
  DataType ex_error;

  rabitq_impl::ex_bits::ex_bits_compact_code(
      data, centroid, padded_dim, ex_bits, cur_ex_data.ex_code(), cur_ex_data.f_add_ex(),
      cur_ex_data.f_rescale_ex(), ex_error, metric_type, config.t_const);
}

// NOLINTBEGIN
template <uint32_t b_query>
inline float warmup_ip_x0_q(
    const uint64_t *data,   // pointer to data blocks (each 64 bits)
    const uint64_t *query,  // pointer to query words (each 64 bits), arranged so that for
                            // each data block the corresponding b_query query words follow
    float delta, float vl, size_t padded_dim,
    [[maybe_unused]] size_t _b_query = 0  // not used
) {
  const size_t num_blk = padded_dim / 64;
  size_t ip_scalar = 0;
  size_t ppc_scalar = 0;

  // Process blocks in chunks of 8
  const size_t vec_width = 8;
  size_t vec_end = (num_blk / vec_width) * vec_width;

  // Vector accumulators (each holds 8 64-bit lanes)
  __m512i ip_vec =
      _mm512_setzero_si512();  // will accumulate weighted popcount intersections per block
  __m512i ppc_vec = _mm512_setzero_si512();  // will accumulate popcounts of data blocks

  // Loop over blocks in batches of 8
  for (size_t i = 0; i < vec_end; i += vec_width) {
    // Load eight 64-bit data blocks into x_vec.
    __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(data + i));

    // Compute popcount for each 64-bit block in x_vec using the AVX512 VPOPCNTDQ
    // instruction. (Ensure you compile with the proper flags for VPOPCNTDQ.)
    __m512i popcnt_x_vec = _mm512_popcnt_epi64(x_vec);
    ppc_vec = _mm512_add_epi64(ppc_vec, popcnt_x_vec);

    // For accumulating the weighted popcounts per block.
    __m512i block_ip = _mm512_setzero_si512();

    // Process each query component (b_query is a compile-time constant, and is small).
    for (uint32_t j = 0; j < b_query; j++) {
      // We need to gather from query array the j-th query for each of the eight
      // blocks. For block (i + k) the index is: ( (i + k) * b_query + j ). We
      // construct an index vector of eight 64-bit indices.
      uint64_t indices[vec_width];
      for (size_t k = 0; k < vec_width; k++) {
        indices[k] = ((i + k) * b_query + j);
      }
      // Load indices from memory.
      __m512i index_vec = _mm512_loadu_si512(indices);
      // Gather 8 query words with a scale of 8 (since query is an array of 64-bit
      // integers).
      __m512i q_vec = _mm512_i64gather_epi64(index_vec, query, 8);

      // Compute bitwise AND of data blocks and corresponding query words.
      __m512i and_vec = _mm512_and_si512(x_vec, q_vec);
      // Compute popcount on each lane.
      __m512i popcnt_and = _mm512_popcnt_epi64(and_vec);

      // Multiply by the weighting factor (1 << j) for this query position.
      const uint64_t shift = 1ULL << j;
      __m512i shift_vec = _mm512_set1_epi64(shift);
      __m512i weighted = _mm512_mullo_epi64(popcnt_and, shift_vec);

      // Accumulate weighted popcounts for these blocks.
      block_ip = _mm512_add_epi64(block_ip, weighted);
    }
    // Add the block's query-weighted popcount to the overall ip vector.
    ip_vec = _mm512_add_epi64(ip_vec, block_ip);
  }

  // Horizontally reduce the vector accumulators.
  uint64_t ip_arr[vec_width];
  uint64_t ppc_arr[vec_width];
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(ip_arr), ip_vec);
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(ppc_arr), ppc_vec);

  for (size_t k = 0; k < vec_width; k++) {
    ip_scalar += ip_arr[k];
    ppc_scalar += ppc_arr[k];
  }

  // Process remaining blocks that did not fit in the vectorized loop.
  for (size_t i = vec_end; i < num_blk; i++) {
    const uint64_t x = data[i];
    ppc_scalar += __builtin_popcountll(x);
    for (uint32_t j = 0; j < b_query; j++) {
      ip_scalar += __builtin_popcountll(x & query[i * b_query + j]) << j;
    }
  }

  return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}
// NOLINTEND

/**
 * @brief 1-bit distance estimation for a single vector (no FastScan)
 */
template <typename DataType, typename IDType>
inline void split_single_estdist(const char *bin_data, const SplitSingleQuery<DataType, IDType> &q_obj,
                          size_t padded_dim, float &ip_x0_qr, float &est_dist, float &low_dist,
                          float g_add = 0, float g_error = 0) {
  ConstBinDataMap<float> cur_bin(bin_data, padded_dim);

  ip_x0_qr = warmup_ip_x0_q<SplitSingleQuery<DataType, IDType>::kNumBits>(
      cur_bin.bin_code(), q_obj.query_bin(), q_obj.delta(), q_obj.vl(), padded_dim,
      SplitSingleQuery<DataType, IDType>::kNumBits);

  est_dist = cur_bin.f_add() + g_add + cur_bin.f_rescale() * (ip_x0_qr + q_obj.k1xsumq());

  low_dist = est_dist - cur_bin.f_error() * g_error;
};

/**
 * @brief Full bits distance estimation for a single vector.
 */
template <typename DataType, typename IDType>
inline void split_single_fulldist(const char *bin_data, const char *ex_data,
                           float (*ip_func_)(const float *, const uint8_t *, size_t),
                           const SplitSingleQuery<DataType, IDType> &q_obj, size_t padded_dim,
                           size_t ex_bits, float &est_dist, float &low_dist, float &ip_x0_qr,
                           float g_add, float g_error) {
  ConstBinDataMap<float> cur_bin(bin_data, padded_dim);
  ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

  // [TODO: optimize this function]
  ip_x0_qr = mask_ip_x0_q(q_obj.rotated_query(), cur_bin.bin_code(), padded_dim);

  est_dist = cur_ex.f_add_ex() + g_add +
             (cur_ex.f_rescale_ex() *
              (static_cast<float>(1 << ex_bits) * ip_x0_qr +
               ip_func_(q_obj.rotated_query(), cur_ex.ex_code(), padded_dim) + q_obj.kbxsumq()));

  low_dist = est_dist - cur_bin.f_error() * g_error / static_cast<float>(1 << ex_bits);
}

/**
   * @brief Use ex-data bits to get more accurate distance
   *
   * @tparam Query
   * @param ex_data ex data, refer to ExDataMap in data_layout.hpp
   * @param ip_func_  inner product function for compactly stored ex codes
   * @param q_obj
   * @param padded_dim
   * @param ex_bits
   * @param ip_x0_qr  intermediate result generated by 1-bit distance estimation
   * @return float
   */
  template <class Query>
  inline auto split_distance_boosting(const char *ex_data,
                               float (*ip_func_)(const float *, const uint8_t *, size_t),
                               const Query &q_obj, size_t padded_dim, size_t ex_bits,
                               float ip_x0_qr) -> float {
    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    float ex_dist =
        cur_ex.f_add_ex() + q_obj.g_add() +
        (cur_ex.f_rescale_ex() *
         (static_cast<float>(1 << ex_bits) * ip_x0_qr +
          ip_func_(q_obj.rotated_query(), cur_ex.ex_code(), padded_dim) + q_obj.kbxsumq()));

    return ex_dist;
  }

}  // namespace alaya