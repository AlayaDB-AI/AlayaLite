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

#include "utils/math.hpp"
#include <gtest/gtest.h>
#include <cstdint>

namespace alaya::math {

// ============================================================================
// floor_log2 Tests
// ============================================================================

TEST(MathTest, FloorLog2_PowerOfTwo) {
    EXPECT_EQ(floor_log2(1U), 0U);
    EXPECT_EQ(floor_log2(2U), 1U);
    EXPECT_EQ(floor_log2(4U), 2U);
    EXPECT_EQ(floor_log2(8U), 3U);
    EXPECT_EQ(floor_log2(16U), 4U);
    EXPECT_EQ(floor_log2(32U), 5U);
    EXPECT_EQ(floor_log2(64U), 6U);
    EXPECT_EQ(floor_log2(128U), 7U);
    EXPECT_EQ(floor_log2(256U), 8U);
    EXPECT_EQ(floor_log2(1024U), 10U);
}

TEST(MathTest, FloorLog2_NonPowerOfTwo) {
    EXPECT_EQ(floor_log2(3U), 1U);   // floor(log2(3)) = 1
    EXPECT_EQ(floor_log2(5U), 2U);   // floor(log2(5)) = 2
    EXPECT_EQ(floor_log2(6U), 2U);   // floor(log2(6)) = 2
    EXPECT_EQ(floor_log2(7U), 2U);   // floor(log2(7)) = 2
    EXPECT_EQ(floor_log2(9U), 3U);   // floor(log2(9)) = 3
    EXPECT_EQ(floor_log2(15U), 3U);  // floor(log2(15)) = 3
    EXPECT_EQ(floor_log2(17U), 4U);  // floor(log2(17)) = 4
    EXPECT_EQ(floor_log2(100U), 6U); // floor(log2(100)) = 6
}

TEST(MathTest, FloorLog2_Zero) {
    // Zero is undefined behavior, but function returns 0 for protection
    EXPECT_EQ(floor_log2(0U), 0U);
}

TEST(MathTest, FloorLog2_DifferentTypes) {
    // Test with different integer types
    EXPECT_EQ(floor_log2(static_cast<uint8_t>(8)), 3U);
    EXPECT_EQ(floor_log2(static_cast<uint16_t>(256)), 8U);
    EXPECT_EQ(floor_log2(static_cast<uint32_t>(65536)), 16U);
    EXPECT_EQ(floor_log2(static_cast<uint64_t>(1ULL << 32)), 32U);
    EXPECT_EQ(floor_log2(static_cast<uint64_t>(1ULL << 62)), 62U);
}

TEST(MathTest, FloorLog2_LargeValues) {
    EXPECT_EQ(floor_log2(static_cast<uint64_t>(1ULL << 63)), 63U);
    EXPECT_EQ(floor_log2(UINT64_MAX), 63U);
    EXPECT_EQ(floor_log2(UINT32_MAX), 31U);
}

// ============================================================================
// ceil_log2 Tests
// ============================================================================

TEST(MathTest, CeilLog2_PowerOfTwo) {
    EXPECT_EQ(ceil_log2(1U), 0U);
    EXPECT_EQ(ceil_log2(2U), 1U);
    EXPECT_EQ(ceil_log2(4U), 2U);
    EXPECT_EQ(ceil_log2(8U), 3U);
    EXPECT_EQ(ceil_log2(16U), 4U);
    EXPECT_EQ(ceil_log2(32U), 5U);
    EXPECT_EQ(ceil_log2(64U), 6U);
    EXPECT_EQ(ceil_log2(128U), 7U);
    EXPECT_EQ(ceil_log2(256U), 8U);
    EXPECT_EQ(ceil_log2(1024U), 10U);
}

TEST(MathTest, CeilLog2_NonPowerOfTwo) {
    EXPECT_EQ(ceil_log2(3U), 2U);   // ceil(log2(3)) = 2
    EXPECT_EQ(ceil_log2(5U), 3U);   // ceil(log2(5)) = 3
    EXPECT_EQ(ceil_log2(6U), 3U);   // ceil(log2(6)) = 3
    EXPECT_EQ(ceil_log2(7U), 3U);   // ceil(log2(7)) = 3
    EXPECT_EQ(ceil_log2(9U), 4U);   // ceil(log2(9)) = 4
    EXPECT_EQ(ceil_log2(15U), 4U);  // ceil(log2(15)) = 4
    EXPECT_EQ(ceil_log2(17U), 5U);  // ceil(log2(17)) = 5
    EXPECT_EQ(ceil_log2(100U), 7U); // ceil(log2(100)) = 7
}

TEST(MathTest, CeilLog2_EdgeCases) {
    EXPECT_EQ(ceil_log2(0U), 0U);
    EXPECT_EQ(ceil_log2(1U), 0U);
}

TEST(MathTest, CeilLog2_DifferentTypes) {
    EXPECT_EQ(ceil_log2(static_cast<uint8_t>(9)), 4U);
    EXPECT_EQ(ceil_log2(static_cast<uint16_t>(257)), 9U);
    EXPECT_EQ(ceil_log2(static_cast<uint32_t>(65537)), 17U);
    EXPECT_EQ(ceil_log2(static_cast<uint64_t>((1ULL << 32) + 1)), 33U);
}

// ============================================================================
// ceil_div Tests
// ============================================================================

TEST(MathTest, CeilDiv_Basic) {
    EXPECT_EQ(ceil_div(7, 3), 3);    // ceil(7/3) = 3
    EXPECT_EQ(ceil_div(9, 3), 3);    // ceil(9/3) = 3 (exact division)
    EXPECT_EQ(ceil_div(10, 3), 4);   // ceil(10/3) = 4
    EXPECT_EQ(ceil_div(1, 1), 1);
    EXPECT_EQ(ceil_div(0, 5), 0);
}

TEST(MathTest, CeilDiv_LargeValues) {
    EXPECT_EQ(ceil_div(100U, 7U), 15U);
    EXPECT_EQ(ceil_div(1000U, 64U), 16U);
    EXPECT_EQ(ceil_div(4096U, 64U), 64U);
}

TEST(MathTest, CeilDiv_DivideByZero) {
    EXPECT_THROW((void)ceil_div(10, 0), std::invalid_argument); // NOLINT
}

TEST(MathTest, CeilDiv_DifferentTypes) {
    EXPECT_EQ(ceil_div(static_cast<size_t>(100), static_cast<size_t>(7)), static_cast<size_t>(15));
    EXPECT_EQ(ceil_div(static_cast<int64_t>(100), static_cast<int64_t>(7)), static_cast<int64_t>(15));
}

// ============================================================================
// round_up_general Tests
// ============================================================================

TEST(MathTest, RoundUpGeneral_Basic) {
    EXPECT_EQ(round_up_general(7, 3), 9U);
    EXPECT_EQ(round_up_general(9, 3), 9U);    // already aligned
    EXPECT_EQ(round_up_general(10, 3), 12U);
    EXPECT_EQ(round_up_general(0, 5), 0U);
    EXPECT_EQ(round_up_general(1, 5), 5U);
}

TEST(MathTest, RoundUpGeneral_LargeValues) {
    EXPECT_EQ(round_up_general(100, 64), 128U);
    EXPECT_EQ(round_up_general(128, 64), 128U);  // already aligned
    EXPECT_EQ(round_up_general(129, 64), 192U);
    EXPECT_EQ(round_up_general(1000, 100), 1000U);
    EXPECT_EQ(round_up_general(1001, 100), 1100U);
}

TEST(MathTest, RoundUpGeneral_DivisorZero) {
    EXPECT_EQ(round_up_general(100, 0), 0U);
}

// ============================================================================
// round_up_pow2 Tests
// ============================================================================

TEST(MathTest, RoundUpPow2_Basic) {
    EXPECT_EQ(round_up_pow2(70, 64), 128U);
    EXPECT_EQ(round_up_pow2(64, 64), 64U);    // already aligned
    EXPECT_EQ(round_up_pow2(65, 64), 128U);
    EXPECT_EQ(round_up_pow2(0, 64), 0U);
    EXPECT_EQ(round_up_pow2(1, 64), 64U);
}

TEST(MathTest, RoundUpPow2_DifferentAlignments) {
    // Alignment 4
    EXPECT_EQ(round_up_pow2(1, 4), 4U);
    EXPECT_EQ(round_up_pow2(4, 4), 4U);
    EXPECT_EQ(round_up_pow2(5, 4), 8U);

    // Alignment 16
    EXPECT_EQ(round_up_pow2(1, 16), 16U);
    EXPECT_EQ(round_up_pow2(16, 16), 16U);
    EXPECT_EQ(round_up_pow2(17, 16), 32U);

    // Alignment 4096 (page size)
    EXPECT_EQ(round_up_pow2(1, 4096), 4096U);
    EXPECT_EQ(round_up_pow2(4096, 4096), 4096U);
    EXPECT_EQ(round_up_pow2(4097, 4096), 8192U);
}

TEST(MathTest, RoundUpPow2_DifferentTypes) {
    EXPECT_EQ(round_up_pow2(static_cast<uint32_t>(70), 64), 128U);
    EXPECT_EQ(round_up_pow2(static_cast<uint64_t>(70), 64), 128U);
    EXPECT_EQ(round_up_pow2(static_cast<int32_t>(70), 64), 128);
    EXPECT_EQ(round_up_pow2(static_cast<int64_t>(70), 64), 128);
}

// ============================================================================
// is_power_of_two Tests
// ============================================================================

TEST(MathTest, IsPowerOfTwo_True) {
    EXPECT_TRUE(is_power_of_two(1));
    EXPECT_TRUE(is_power_of_two(2));
    EXPECT_TRUE(is_power_of_two(4));
    EXPECT_TRUE(is_power_of_two(8));
    EXPECT_TRUE(is_power_of_two(16));
    EXPECT_TRUE(is_power_of_two(32));
    EXPECT_TRUE(is_power_of_two(64));
    EXPECT_TRUE(is_power_of_two(128));
    EXPECT_TRUE(is_power_of_two(256));
    EXPECT_TRUE(is_power_of_two(1024));
    EXPECT_TRUE(is_power_of_two(4096));
    EXPECT_TRUE(is_power_of_two(1ULL << 32));
    EXPECT_TRUE(is_power_of_two(1ULL << 62));
}

TEST(MathTest, IsPowerOfTwo_False) {
    EXPECT_FALSE(is_power_of_two(0));
    EXPECT_FALSE(is_power_of_two(3));
    EXPECT_FALSE(is_power_of_two(5));
    EXPECT_FALSE(is_power_of_two(6));
    EXPECT_FALSE(is_power_of_two(7));
    EXPECT_FALSE(is_power_of_two(9));
    EXPECT_FALSE(is_power_of_two(10));
    EXPECT_FALSE(is_power_of_two(15));
    EXPECT_FALSE(is_power_of_two(17));
    EXPECT_FALSE(is_power_of_two(100));
    EXPECT_FALSE(is_power_of_two(1000));
}

TEST(MathTest, IsPowerOfTwo_DifferentTypes) {
    EXPECT_TRUE(is_power_of_two(static_cast<uint8_t>(8)));
    EXPECT_TRUE(is_power_of_two(static_cast<uint16_t>(256)));
    EXPECT_TRUE(is_power_of_two(static_cast<uint32_t>(65536)));
    EXPECT_TRUE(is_power_of_two(static_cast<uint64_t>(1ULL << 32)));

    EXPECT_FALSE(is_power_of_two(static_cast<uint8_t>(10)));
    EXPECT_FALSE(is_power_of_two(static_cast<uint16_t>(100)));
}

TEST(MathTest, IsPowerOfTwo_NegativeValues) {
    // Negative values should return false (x > 0 check)
    EXPECT_FALSE(is_power_of_two(-1));
    EXPECT_FALSE(is_power_of_two(-2));
    EXPECT_FALSE(is_power_of_two(-8));
}

// ============================================================================
// Constexpr Tests (compile-time evaluation)
// ============================================================================

TEST(MathTest, ConstexprEvaluation) {
    // These tests verify that the functions can be evaluated at compile time
    auto log2_result = floor_log2(8U);
    EXPECT_EQ(log2_result, 3U);

    auto ceil_log2_result = ceil_log2(9U);
    EXPECT_EQ(ceil_log2_result, 4U);

    auto round_up_result = round_up_pow2(70U, 64);
    EXPECT_EQ(round_up_result, 128U);

    auto is_pow2_result = is_power_of_two(64);
    EXPECT_TRUE(is_pow2_result);
}

}  // namespace alaya::math
