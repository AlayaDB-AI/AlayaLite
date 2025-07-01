#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <limits>

#define BIT_ID(x) (__builtin_popcount((x) - 1))
#define LOWBIT(x) ((x) & (-(x)))

namespace alaya {
using PID = uint32_t;

constexpr uint32_t kPidMax = 0xFFFFFFFF;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;

template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;

template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using RowMajorArrayMap = Eigen::Map<RowMajorArray<T>>;

template <typename T>
using ConstRowMajorArrayMap = Eigen::Map<const RowMajorArray<T>>;

template <typename T>
using VectorMap = Eigen::Map<Vector<T>>;

template <typename T>
using ConstVectorMap = Eigen::Map<const Vector<T>>;

template <typename T, typename TP = PID>
struct AnnCandidate {
  TP id = 0;                                   // NOLINT
  T distance = std::numeric_limits<T>::max();  // NOLINT

  AnnCandidate() = default;
  explicit AnnCandidate(TP vec_id, T dis) : id(vec_id), distance(dis) {}

  friend auto operator<(const AnnCandidate &first, const AnnCandidate &second) -> bool {
    return first.distance < second.distance;
  }
  friend auto operator>(const AnnCandidate &first, const AnnCandidate &second) -> bool {
    return first.distance > second.distance;
  }
  friend auto operator>=(const AnnCandidate &first, const AnnCandidate &second) -> bool {
    return first.distance >= second.distance;
  }
  friend auto operator<=(const AnnCandidate &first, const AnnCandidate &second) -> bool {
    return first.distance <= second.distance;
  }
};

// NOLINTBEGIN
enum class ScalarQuantizerType : std::uint8_t { RECONSTRUCTION, UNBIASED_ESTIMATION, PLAIN };
// NOLINTEND

}  // namespace alaya