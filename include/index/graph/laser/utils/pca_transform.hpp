// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace alaya::laser {

/**
 * @brief PCA Transform class for online query transformation.
 *
 * This class loads pre-trained PCA parameters (mean and principal components)
 * and applies the transformation to query vectors at search time.
 *
 * Note: This is an orthogonal transformation (not dimensionality reduction),
 * which reorders dimensions by variance in descending order.
 */
class PCATransform {
 private:
  size_t input_dim_ = 0;           // Input vector dimension
  size_t output_dim_ = 0;          // Output vector dimension (typically same as input)
  std::vector<float> pca_matrix_;  // [output_dim_ * input_dim_]
  std::vector<float> mean_;        // [input_dim_]
  bool loaded_ = false;

 public:
  PCATransform() = default;

  /**
   * @brief Constructor with dimensions.
   * @param input_dim Input dimension
   * @param output_dim Output dimension (default 0 means same as input)
   */
  explicit PCATransform(size_t input_dim, size_t output_dim = 0)
      : input_dim_(input_dim), output_dim_(output_dim == 0 ? input_dim : output_dim) {
    pca_matrix_.resize(output_dim_ * input_dim_);
    mean_.resize(input_dim_);
  }

  /**
   * @brief Train PCA on data.
   * @param data Pointer to training data (count * input_dim floats, row-major)
   * @param count Number of training vectors
   */
  void train(const float *data, uint64_t count) {
    std::vector<float> centralized_data(count * input_dim_, 0.0F);
    std::vector<float> covariance_matrix(input_dim_ * input_dim_, 0.0F);

    // 1. Compute mean (stored in mean_)
    compute_column_mean(data, count);

    // 2. Centralize data
    for (uint64_t i = 0; i < count; ++i) {
      centralize_data(data + i * input_dim_, centralized_data.data() + i * input_dim_);
    }

    // 3. Get covariance matrix
    compute_covariance_matrix(centralized_data.data(), count, covariance_matrix.data());

    // 4. Eigen decomposition (stored in pca_matrix_)
    perform_eigen_decomposition(covariance_matrix.data());

    loaded_ = true;
  }

  /**
   * @brief Load PCA parameters from binary file.
   *
   * File format:
   *   - uint64_t: dimension
   *   - float[dim]: mean vector
   *   - float[dim * dim]: components matrix (row-major)
   *
   * @param filename Path to the PCA parameters file
   * @return true if loaded successfully, false otherwise
   */
  bool load(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
      std::cerr << "PCATransform: Cannot open file " << filename << std::endl;
      return false;
    }

    // Read dimension
    uint64_t dim;
    input.read(reinterpret_cast<char *>(&dim), sizeof(uint64_t));
    input_dim_ = static_cast<size_t>(dim);
    output_dim_ = input_dim_;  // Orthogonal transformation

    // Resize vectors
    mean_.resize(input_dim_);
    pca_matrix_.resize(output_dim_ * input_dim_);

    // Read mean vector
    input.read(reinterpret_cast<char *>(mean_.data()),
               static_cast<std::streamsize>(sizeof(float) * input_dim_));

    // Read components matrix (row-major in file)
    input.read(reinterpret_cast<char *>(pca_matrix_.data()),
               static_cast<std::streamsize>(sizeof(float) * output_dim_ * input_dim_));

    input.close();
    loaded_ = true;

    std::cout << "PCATransform: Loaded PCA parameters, dim = " << input_dim_ << std::endl;
    return true;
  }

  /**
   * @brief Save PCA parameters to binary file.
   * @param filename Path to save the PCA parameters
   * @return true if saved successfully, false otherwise
   */
  bool save(const std::string &filename) const {
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
      std::cerr << "PCATransform: Cannot open file for writing " << filename << std::endl;
      return false;
    }

    // Write dimension
    uint64_t dim = static_cast<uint64_t>(input_dim_);
    output.write(reinterpret_cast<const char *>(&dim), sizeof(uint64_t));

    // Write mean vector
    output.write(reinterpret_cast<const char *>(mean_.data()),
                 static_cast<std::streamsize>(sizeof(float) * input_dim_));

    // Write components matrix
    output.write(reinterpret_cast<const char *>(pca_matrix_.data()),
                 static_cast<std::streamsize>(sizeof(float) * output_dim_ * input_dim_));

    output.close();
    return true;
  }

  /**
   * @brief Transform a single vector using PCA.
   *
   * Computes: output = pca_matrix * (input - mean)
   *
   * @param input Pointer to input vector (input_dim floats)
   * @param output Pointer to output vector (output_dim floats)
   */
  void transform(const float *input, float *output) const {
    if (!loaded_) {
      std::cerr << "PCATransform: PCA parameters not loaded!" << std::endl;
      return;
    }

    // Centralize input
    std::vector<float> centralized(input_dim_);
    centralize_data(input, centralized.data());

    // Matrix-vector multiplication: output = pca_matrix * centralized
    // pca_matrix is [output_dim_ x input_dim_] in row-major
    // Use Eigen for efficient computation
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        pca_map(pca_matrix_.data(),
                static_cast<Eigen::Index>(output_dim_),
                static_cast<Eigen::Index>(input_dim_));
    Eigen::Map<const Eigen::VectorXf> centered_vec(centralized.data(),
                                                   static_cast<Eigen::Index>(input_dim_));
    Eigen::Map<Eigen::VectorXf> result(output, static_cast<Eigen::Index>(output_dim_));

    result.noalias() = pca_map * centered_vec;
  }

  /**
   * @brief Inverse transform (from PCA space back to original space).
   * @param input Pointer to input vector in PCA space (output_dim floats)
   * @param output Pointer to output vector in original space (input_dim floats)
   */
  void inverse_transform(const float *input, float *output) const {
    if (!loaded_) {
      std::cerr << "PCATransform: PCA parameters not loaded!" << std::endl;
      return;
    }

    // output = pca_matrix^T * input + mean
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        pca_map(pca_matrix_.data(),
                static_cast<Eigen::Index>(output_dim_),
                static_cast<Eigen::Index>(input_dim_));
    Eigen::Map<const Eigen::VectorXf> input_vec(input, static_cast<Eigen::Index>(output_dim_));
    Eigen::Map<Eigen::VectorXf> result(output, static_cast<Eigen::Index>(input_dim_));
    Eigen::Map<const Eigen::VectorXf> mean_vec(mean_.data(), static_cast<Eigen::Index>(input_dim_));

    result.noalias() = pca_map.transpose() * input_vec + mean_vec;
  }

  [[nodiscard]] size_t dimension() const { return input_dim_; }
  [[nodiscard]] size_t input_dimension() const { return input_dim_; }
  [[nodiscard]] size_t output_dimension() const { return output_dim_; }
  [[nodiscard]] bool is_loaded() const { return loaded_; }

  // Test helper methods
  void copy_pca_matrix_for_test(float *out_pca_matrix) const {
    std::memcpy(out_pca_matrix, pca_matrix_.data(), pca_matrix_.size() * sizeof(float));
  }

  void copy_mean_for_test(float *out_mean) const {
    std::memcpy(out_mean, mean_.data(), mean_.size() * sizeof(float));
  }

  void set_mean_for_test(const float *input_mean) {
    std::memcpy(mean_.data(), input_mean, mean_.size() * sizeof(float));
  }

  void set_pca_matrix_for_test(const float *input_pca_matrix) {
    std::memcpy(pca_matrix_.data(), input_pca_matrix, pca_matrix_.size() * sizeof(float));
  }

 private:
  void compute_column_mean(const float *data, uint64_t count) {
    std::fill(mean_.begin(), mean_.end(), 0.0F);

    for (uint64_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < input_dim_; ++j) {
        mean_[j] += data[i * input_dim_ + j];
      }
    }

    for (size_t j = 0; j < input_dim_; ++j) {
      mean_[j] /= static_cast<float>(count);
    }
  }

  void centralize_data(const float *original_data, float *centralized_data) const {
    for (size_t j = 0; j < input_dim_; ++j) {
      centralized_data[j] = original_data[j] - mean_[j];
    }
  }

  void compute_covariance_matrix(const float *centralized_data,
                                 uint64_t count,
                                 float *covariance_matrix) const {
    // Use Eigen for efficient matrix multiplication: cov = X^T * X / (n-1)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_map(centralized_data,
                 static_cast<Eigen::Index>(count),
                 static_cast<Eigen::Index>(input_dim_));
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        cov_map(covariance_matrix,
                static_cast<Eigen::Index>(input_dim_),
                static_cast<Eigen::Index>(input_dim_));

    cov_map.noalias() = data_map.transpose() * data_map;

    // Unbiased estimator
    float scale = 1.0F / static_cast<float>(count - 1);
    cov_map *= scale;
  }

  bool perform_eigen_decomposition(const float *covariance_matrix) {
    // Use Eigen's SelfAdjointEigenSolver for symmetric matrix
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        cov_map(covariance_matrix,
                static_cast<Eigen::Index>(input_dim_),
                static_cast<Eigen::Index>(input_dim_));

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov_map);
    if (solver.info() != Eigen::Success) {
      std::cerr << "PCATransform: Eigen decomposition failed" << std::endl;
      return false;
    }

    // Eigenvalues are in ascending order, we need descending
    // So we take eigenvectors in reverse order
    const Eigen::MatrixXf &eigen_vectors = solver.eigenvectors();

    // pca_matrix_[i][input_dim_] = eigen_vectors[:, input_dim_ - 1 - i]
    for (size_t i = 0; i < output_dim_; ++i) {
      for (size_t j = 0; j < input_dim_; ++j) {
        pca_matrix_[i * input_dim_ + j] =
            eigen_vectors(static_cast<Eigen::Index>(j),
                          static_cast<Eigen::Index>(input_dim_ - 1 - i));
      }
    }
    return true;
  }
};

}  // namespace alaya::laser
