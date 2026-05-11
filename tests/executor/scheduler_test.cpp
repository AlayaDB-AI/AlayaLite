// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "executor/scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace alaya {

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpus_ = {0};
    scheduler_ = std::make_unique<Scheduler>(cpus_);
  }

  void TearDown() override { scheduler_->join(); }

  std::vector<CpuID> cpus_;
  std::unique_ptr<Scheduler> scheduler_;
};

}  // namespace alaya
