#pragma once

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>

#include "../quant/rbq.hpp"
#include "../space_concepts.hpp"
#include "rotator.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/rbq_utils/defines.hpp"
#include "utils/rbq_utils/rbq_impl.hpp"
#include "utils/rbq_utils/space.hpp"

namespace alaya {

template <typename DataType = float, typename IDType = PID, typename DistanceType = float>
class RBQSpace {
  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;

 private:
  MetricType metric_{MetricType::L2};
  uint32_t dim_{0};
  std::unique_ptr<Rotator<DataType>> rotator_ = nullptr;
  RotatorType rotator_type_ = RotatorType::FhtKacRotator;
  DistFunc<DataType, DistanceType> distance_calu_func_;

  auto set_metric_function() -> void {
    if (metric_ == MetricType::NONE) {
      metric_ = MetricType::L2;
    }
    switch (metric_) {
      case MetricType::L2:
        distance_calu_func_ = euclidean_sqr<DataType>;
        break;
      case MetricType::COS:
      case MetricType::IP:
        distance_calu_func_ = dot_product_dis<DataType>;
        break;
      default:
        break;
    }
  }

 public:
  RBQSpace() = default;
  ~RBQSpace() = default;
  RBQSpace(size_t dim, MetricType metric, RotatorType rt)
      : dim_(dim), metric_(metric), rotator_type_(rt) {
    rotator_ = choose_rotator<DataType>(dim, rotator_type_, round_up_to_multiple(dim_, 64));
    set_metric_function();
  }

  auto operator=(const RBQSpace &other) -> RBQSpace & = default;
  auto operator=(const RBQSpace &&other) noexcept -> RBQSpace & = default;

  auto get_metric_type() -> MetricType { return metric_; }

  auto get_dist_func() -> DistFunc<DataType, DistanceType> { return distance_calu_func_; }

  auto get_distance(DataType *v1, DataType *v2, size_t pass_dim) -> DistanceType {
    return distance_calu_func_(v1, v2, pass_dim);
  }

  auto get_bin_est(std::vector<DistanceType> &q_to_centroids,
                   SplitSingleQuery<DataType, IDType> &query_wrapper, EstimateRecord &res,
                   size_t num_cluster, IDType cluster_id /*get_clusterid_by_internalid(currObj)*/,
                   char *bin_data /*get_bindata_by_internalid(currObj)*/
                   ) -> void {
    if (metric_ == MetricType::IP || metric_ == MetricType::COS) {
      float norm = q_to_centroids[cluster_id];
      float error = q_to_centroids[cluster_id + num_cluster];
      split_single_estdist(bin_data, query_wrapper, get_rotator_size(), res.ip_x0_qr, res.est_dist,
                           res.low_dist, -norm, error);
    } else {  // l2 distance
      float norm = q_to_centroids[cluster_id];
      split_single_estdist(bin_data, query_wrapper, get_rotator_size(), res.ip_x0_qr, res.est_dist,
                           res.low_dist, norm * norm, norm);
    }
  }

  auto get_ex_est(std::vector<DistanceType> &q_to_centroids,
                  SplitSingleQuery<DataType, IDType> &query_wrapper, EstimateRecord &res,
                  size_t ex_bits, IDType cluster_id /*get_clusterid_by_internalid(currObj)*/,
                  char *ex_data /*get_exdata_by_internalid(currObj)*/
                  ) -> void {
    query_wrapper.set_g_add(q_to_centroids[cluster_id]);
    float est_dist = split_distance_boosting(ex_data, select_excode_ipfunc(ex_bits), query_wrapper,
                                             get_rotator_size(), ex_bits, res.ip_x0_qr);
    float low_dist = est_dist - (res.est_dist - res.low_dist) / (1 << ex_bits);  // NOLINT
    res.est_dist = est_dist;
    res.low_dist = low_dist;
    // Note that res.ip_x0_qr becomes invalid after this function.
  }

  auto get_full_est(std::vector<DistanceType> &q_to_centroids,
                    SplitSingleQuery<DataType, IDType> &query_wrapper, EstimateRecord &res,
                    size_t num_cluster, size_t ex_bits,
                    IDType cluster_id /*get_clusterid_by_internalid(currObj)*/,
                    char *bin_data /*get_bindata_by_internalid(currObj)*/,
                    char *ex_data /*get_exdata_by_internalid(currObj)*/
  ) const -> void {
    if (metric_ == MetricType::IP || metric_ == MetricType::COS) {
      float norm = q_to_centroids[cluster_id];
      float error = q_to_centroids[cluster_id + num_cluster];
      split_single_fulldist(bin_data, ex_data, select_excode_ipfunc(ex_bits), query_wrapper,
                            get_rotator_size(), ex_bits, res.est_dist, res.low_dist, res.ip_x0_qr,
                            -norm, error);
    } else {  // L2 distance
      float norm = q_to_centroids[cluster_id];
      split_single_fulldist(bin_data, ex_data, select_excode_ipfunc(ex_bits), query_wrapper,
                            get_rotator_size(), ex_bits, res.est_dist, res.low_dist, res.ip_x0_qr,
                            norm * norm, norm);
    }
  }

  void quantize_split_single(const DataType *data, const DataType *centroid, size_t ex_bits,
                             char *bin_data, char *ex_data,
                             rabitq_impl::RabitqConfig config = rabitq_impl::RabitqConfig()) {
    quantize_compact_one_bit(data, centroid, get_rotator_size(), bin_data, metric_);
    if (ex_bits > 0) {
      quantize_compact_ex_bits(data, centroid, get_rotator_size(), ex_bits, ex_data, metric_,
                               config);
    }
  }

  auto load(std::ifstream &reader) -> void {
    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&rotator_type_), sizeof(rotator_type_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    rotator_ = choose_rotator<DataType>(dim_, rotator_type_, round_up_to_multiple(dim_, 64));
    set_metric_function();
    // it is possible that two padded_dim are different?
    rotator_->load(reader);
    LOG_INFO("RBQSpace is loaded.");
  }

  auto save(std::ofstream &writer) -> void {
    writer.write(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<char *>(&rotator_type_), sizeof(rotator_type_));
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));

    rotator_->save(writer);
    LOG_INFO("RBQSpace is saved.");
  }

  auto get_rotator_size() -> size_t { return rotator_->size(); }

  auto rotate_data(const DataType *src, DataType *dst) -> void { rotator_->rotate(src, dst); }
};

}  // namespace alaya
