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

template <typename DataType>
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
    rabitq_impl::total_bits::rabitq_scalar_impl<DataType, uint16_t>(
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

template <uint32_t b_query>
inline auto warmup_ip_x0_q(const uint64_t *data, const uint64_t *query, float delta, float vl,
                           size_t padded_dim = 0) -> float {
  auto num_blk = padded_dim / 64;
  const auto *it_data = data;
  const auto *it_query = query;

  size_t ip = 0;
  size_t ppc = 0;

  for (size_t i = 0; i < num_blk; ++i) {
    uint64_t x = *static_cast<const uint64_t *>(it_data);
    ppc += __builtin_popcountll(x);

    for (size_t j = 0; j < b_query; ++j) {
      uint64_t y = *static_cast<const uint64_t *>(it_query);
      ip += (__builtin_popcountll(x & y) << j);
      it_query++;
    }
    it_data++;
  }

  return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

/**
 * @brief 1-bit distance estimation for a single vector (no FastScan)
 */
template <typename DataType>
inline void split_single_estdist(const char *bin_data,
                                 const SplitSingleQuery<DataType> &q_obj, size_t padded_dim,
                                 float &ip_x0_qr, float &est_dist, float &low_dist, float g_add = 0,
                                 float g_error = 0) {
  ConstBinDataMap<float> cur_bin(bin_data, padded_dim);

  ip_x0_qr = warmup_ip_x0_q<SplitSingleQuery<DataType>::kNumBits>(
      cur_bin.bin_code(), q_obj.query_bin(), q_obj.delta(), q_obj.vl(), padded_dim);

  est_dist = cur_bin.f_add() + g_add + cur_bin.f_rescale() * (ip_x0_qr + q_obj.k1xsumq());
  low_dist = est_dist - cur_bin.f_error() * g_error;
};

/**
 * @brief Full bits distance estimation for a single vector.
 */
template <typename DataType>
inline void split_single_fulldist(const char *bin_data, const char *ex_data,
                                  float (*ip_func_)(const float *, const uint8_t *, size_t),
                                  const SplitSingleQuery<DataType> &q_obj,
                                  size_t padded_dim, size_t ex_bits, float &est_dist,
                                  float &low_dist, float &ip_x0_qr, float g_add, float g_error) {
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