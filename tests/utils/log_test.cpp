// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/log.hpp"
#include <gtest/gtest.h>

namespace alaya {
TEST(LogTest, Show) {
  LOG_TRACE("test {}", "tracing");
  LOG_DEBUG("test {}", "debug");
  LOG_INFO("test {}", "info");
  LOG_WARN("test {}", "warn");
  LOG_ERROR("test {}", "error");
  LOG_CRITICAL("test {}", "critical");

  EXPECT_EQ(0, 0);
}
}  // namespace alaya
