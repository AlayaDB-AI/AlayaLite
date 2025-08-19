#pragma once

#include <fmt/core.h>
#include <cstddef>
#include <cstdint>
#include "utils/log.hpp"
#include "utils/rbq_utils/fastscan.hpp"
#include "utils/rbq_utils/rotator.hpp"

namespace alaya {
template <typename DataType>
struct RBQQuantizer {
 private:
  uint32_t dim_{0};                               ///< dimension
  uint32_t padded_dim_{0};                        ///< padded dimension
  RotatorType type_{RotatorType::FhtKacRotator};  ///< rotator type
  std::unique_ptr<Rotator<DataType>> rotator_;    ///< data rotator

  /**
   * @brief pack 0/1 uncompacted integer data (binary_code) to compacted bytes data (compact_code)
   *
   * @param binary_code uncompact quantization code, e.g., [1,0,1,0,0,1,1,0]
   * @param compact_code compact quantization code, e.g., [10100110]
   */
  void pack_binary(const int *__restrict__ binary_code, uint8_t *__restrict__ compact_code) {
    constexpr size_t kTypeBits = sizeof(uint8_t) * 8;
    // i points to the start point of each batch
    for (size_t i = 0; i < padded_dim_; i += kTypeBits) {
      uint8_t cur = 0;
      // j represents offset within a byte,j∈[0,kTypeBits-1]
      for (size_t j = 0; j < kTypeBits; ++j) {
        cur |= (static_cast<uint8_t>(binary_code[i + j]) << (kTypeBits - 1 - j));
      }
      *compact_code = cur;
      // 1 byte(8 bits) each batch
      ++compact_code;
    }
  }

  /**
   * @brief Calculate factors(f_add and f_scale) and quantization code for one neighbor
   *
   * @param data Roatated neighbor data, len: padded_dim
   * @param centroid Rotated centroid data pointer, len: padded_dim
   * @param binary_code Store uncompacted quantization code, every int is either 1 or 0
   * @param f_add One of factors
   * @param f_rescale One of factors
   */
  void cal_fac_and_qc(const DataType *data, const DataType *centroid, int *binary_code,
                      DataType &f_add, DataType &f_rescale) {
    // map pointer to array
    ConstRowMajorArrayMap<DataType> data_arr(data, 1, padded_dim_);
    ConstRowMajorArrayMap<DataType> cent_arr(centroid, 1, padded_dim_);
    // P^(-1)·（o_r - c）
    RowMajorArray<DataType> residual_arr = data_arr - cent_arr;
    // |P^(-1)·(or-c)|^2 = |or-c|^2 (Orthogonal transformations preserve Euclidean distance)
    DataType l2_sqr = l2norm_sqr<DataType>(residual_arr.data(), padded_dim_);

    // unsigned representation, modifications to y_u will be cast to binary code as well
    RowMajorArrayMap<int> y_u(binary_code, 1, static_cast<long>(padded_dim_));  // NOLINT
    // calculate quantization code
    y_u = (residual_arr > 0).template cast<int>();

    float cb = -((1 << 1) - 1) / 2.F;
    // y_bar = y_u + c_b * 1_D
    RowMajorArray<DataType> yu_cb = y_u.template cast<DataType>() + cb;
    // dot product between residual and xu_cb, i.e.,<y_bar,P^(-1)·(or-c)>
    DataType ip_resi_xucb = dot_product<DataType>(residual_arr.data(), yu_cb.data(), padded_dim_);
    // dot product between centroid and xu_cb, i.e.,<y_bar,P^(-1)·c>
    DataType ip_cent_xucb = dot_product<DataType>(centroid, yu_cb.data(), padded_dim_);
    if (ip_resi_xucb == 0) {
      ip_resi_xucb = std::numeric_limits<DataType>::infinity();
    }

    // calculate factors (for l2 metric type only)
    f_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_resi_xucb;
    f_rescale = -2 * l2_sqr / ip_resi_xucb;
  }

 public:
  RBQQuantizer() = default;
  ~RBQQuantizer() = default;

  RBQQuantizer(const RBQQuantizer &) = delete;
  auto operator=(const RBQQuantizer &) -> RBQQuantizer & = delete;

  RBQQuantizer(const RBQQuantizer &&) = delete;
  auto operator=(const RBQQuantizer &&) -> RBQQuantizer & = delete;

  explicit RBQQuantizer(const uint32_t &dim, RotatorType type = RotatorType::FhtKacRotator)
      : dim_(dim), type_(type) {
    rotator_ = choose_rotator<DataType>(dim_, type_, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();
  }

  auto get_padded_dim() const -> uint32_t { return padded_dim_; }

  // dim -> padded dim
  auto rotate_vec(const DataType *src, DataType *dst) const -> void { rotator_->rotate(src, dst); }

  // use one vertex as the centroid and quantize its neighbors
  auto batch_quantize(const DataType *neighbors /* len: num * dim */,
                      const DataType *centroid /* single centroid, len: dim */,
                      size_t num /* total number of the neighbors in this batch */,
                      /* The following pointers point to where the result data is stored */
                      uint8_t *bin_code, DataType *f_add, DataType *f_rescale) -> void {
    // rotate centroid and its neighbors for later processing
    std::vector<DataType> rotated_data((num * padded_dim_));
    std::vector<DataType> rotated_centroid(padded_dim_);
    for (size_t i = 0; i < num; ++i) {
      rotator_->rotate(neighbors + (i * dim_), &rotated_data[i * padded_dim_]);
    }
    rotator_->rotate(centroid, rotated_centroid.data());

    // for compacted quantization code storage
    std::vector<uint8_t> compact_codes(num * padded_dim_ / 8);  // 1 bit/dim

    /// todo: parallelable?
    for (size_t i = 0; i < num; ++i) {  // ith neighbor
      auto nei = reinterpret_cast<DataType *>(rotated_data.data()) + (i * padded_dim_);

      // for uncompacted quantization code storage
      std::vector<int> binary_code(padded_dim_);
      cal_fac_and_qc(nei, rotated_centroid.data(), binary_code.data(), f_add[i], f_rescale[i]);

      // the number of bits in every uint8_t
      constexpr size_t kTypeBits = 8;
      // padded_dim_ / kTypeBits denotes the total number of uint8_t needed for a single neighbor's
      // quantization code
      auto compact_code =
          reinterpret_cast<uint8_t *>(compact_codes.data()) + (padded_dim_ / kTypeBits * i);
      // pack 0/1 uncompacted integer data (binary_code) to compacted bytes data (compact_code)
      pack_binary(binary_code.data(), compact_code);
    }

    // restructure quantization codes for later fastscan computation in querying phase
    fastscan::pack_codes(padded_dim_, compact_codes.data(), num, bin_code);
  }

  auto save(std::ofstream &writer) -> void {
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&padded_dim_), sizeof(padded_dim_));
    writer.write(reinterpret_cast<char *>(&type_), sizeof(type_));

    rotator_->save(writer);

    LOG_INFO("rabitq quantizer is saved.");
  }

  auto load(std::ifstream &reader) -> void {
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&padded_dim_), sizeof(padded_dim_));
    reader.read(reinterpret_cast<char *>(&type_), sizeof(type_));

    rotator_ = choose_rotator<DataType>(dim_, type_, round_up_to_multiple(dim_, 64));
    rotator_->load(reader);

    LOG_INFO("rabitq quantizer is loaded.");
  }
};
};  // namespace alaya