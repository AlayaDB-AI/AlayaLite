/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "index/graph/vamana/build_dispatch.hpp"

namespace {

alaya::vamana::BuildVamanaParams valid_params_until_file_read() {
  alaya::vamana::BuildVamanaParams params =
      alaya::vamana::kDefaultVamanaBuildParams;
  params.data_path = "/tmp/alayalite-vamana-missing.fbin";
  params.output_path = "/tmp/alayalite-vamana-out.index";
  params.R = 64;
  params.L = 100;
  return params;
}

void expect_invalid_sampling_rate(float sampling_rate) {
  auto params = valid_params_until_file_read();
  params.sampling_rate = sampling_rate;
  try {
    alaya::vamana::build_vamana(params);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("sampling_rate"), std::string::npos);
  }
}

TEST(VamanaBuildDispatchTest, RejectsExplicitZeroSamplingRate) {
  expect_invalid_sampling_rate(0.0F);
}

TEST(VamanaBuildDispatchTest, RejectsSamplingRateAboveOne) {
  expect_invalid_sampling_rate(1.1F);
}

}  // namespace
