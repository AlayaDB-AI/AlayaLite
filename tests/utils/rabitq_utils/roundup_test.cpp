/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

#include "utils/rabitq_utils/roundup.hpp"

namespace alaya {

// Test floor_log2 function
TEST(MathUtilsTest, FloorLog2) {
  // Edge cases: x = 0 and x = 1
  EXPECT_EQ(floor_log2(0), 0);  // 0 >> 1 is still 0, loop doesn't execute
  EXPECT_EQ(floor_log2(1), 0);  // 1 > 1 is false, loop doesn't execute

  // Powers of 2
  EXPECT_EQ(floor_log2(2), 1);      // 2^1
  EXPECT_EQ(floor_log2(4), 2);      // 2^2
  EXPECT_EQ(floor_log2(8), 3);      // 2^3
  EXPECT_EQ(floor_log2(16), 4);     // 2^4
  EXPECT_EQ(floor_log2(1024), 10);  // 2^10

  // Non-powers of 2
  EXPECT_EQ(floor_log2(3), 1);     // floor(log2(3)) = 1
  EXPECT_EQ(floor_log2(5), 2);     // floor(log2(5)) = 2
  EXPECT_EQ(floor_log2(7), 2);     // floor(log2(7)) = 2
  EXPECT_EQ(floor_log2(9), 3);     // floor(log2(9)) = 3
  EXPECT_EQ(floor_log2(1000), 9);  // 2^9 = 512, 2^10 = 1024
}

// Test ceil_log2 function
TEST(MathUtilsTest, CeilLog2) {
  // Edge cases
  EXPECT_EQ(ceil_log2(0), 0);  // floor_log2(0) = 0, (1 << 0) = 1, 1 < 0 is false → return 0
  EXPECT_EQ(ceil_log2(1), 0);  // floor_log2(1) = 0, (1 << 0) = 1, 1 < 1 is false → return 0

  // Powers of 2 (should equal floor_log2)
  EXPECT_EQ(ceil_log2(2), 1);   // 2^1
  EXPECT_EQ(ceil_log2(4), 2);   // 2^2
  EXPECT_EQ(ceil_log2(8), 3);   // 2^3
  EXPECT_EQ(ceil_log2(16), 4);  // 2^4

  // Non-powers of 2 (should equal floor_log2 + 1)
  EXPECT_EQ(ceil_log2(3), 2);      // ceil(log2(3)) = 2
  EXPECT_EQ(ceil_log2(5), 3);      // ceil(log2(5)) = 3
  EXPECT_EQ(ceil_log2(7), 3);      // ceil(log2(7)) = 3
  EXPECT_EQ(ceil_log2(9), 4);      // ceil(log2(9)) = 4
  EXPECT_EQ(ceil_log2(1000), 10);  // ceil(log2(1000)) = 10 (since 2^10 = 1024 > 1000)
}

// Test ceil_round_up function with valid inputs
TEST(MathUtilsTest, CeilRoundUp_ValidInputs) {
  // Exact division cases
  EXPECT_EQ(ceil_round_up(10, 5), 2);
  EXPECT_EQ(ceil_round_up(0, 1), 0);
  EXPECT_EQ(ceil_round_up(100, 10), 10);

  // Non-exact division cases
  EXPECT_EQ(ceil_round_up(11, 5), 3);  // 11/5 = 2, remainder = 1 → 2 + 1 = 3
  EXPECT_EQ(ceil_round_up(1, 2), 1);   // 1/2 = 0, remainder = 1 → 0 + 1 = 1
  EXPECT_EQ(ceil_round_up(7, 3), 3);   // 7/3 = 2, remainder = 1 → 2 + 1 = 3

  // Edge cases
  EXPECT_EQ(ceil_round_up(1, 1), 1);
  EXPECT_EQ(ceil_round_up(static_cast<size_t>(10), static_cast<size_t>(3)), 4);
}

// Test ceil_round_up function with invalid divisor (zero or negative)
TEST(MathUtilsTest, CeilRoundUp_InvalidDivisor) {
  // Test zero divisor
  EXPECT_THROW(ceil_round_up(10, 0), std::invalid_argument);

  // Test negative divisor (for signed types)
  EXPECT_THROW(ceil_round_up(10, -1), std::invalid_argument);
  EXPECT_THROW(ceil_round_up(10, -5), std::invalid_argument);

  // Test with different signed integer types
  EXPECT_THROW(ceil_round_up(static_cast<int>(10), static_cast<int>(0)), std::invalid_argument);
  EXPECT_THROW(ceil_round_up(static_cast<long>(10), static_cast<long>(-1)), std::invalid_argument);
}

// Test type safety: ceil_round_up only accepts integral types
TEST(MathUtilsTest, CeilRoundUpTypeSafety) {
  // Compile-time check: the following should compile successfully (integral types)
  EXPECT_EQ(ceil_round_up(static_cast<int>(10), static_cast<int>(3)), 4);
  EXPECT_EQ(ceil_round_up(static_cast<long>(10), static_cast<long>(3)), 4);
  EXPECT_EQ(ceil_round_up(static_cast<unsigned>(10), static_cast<unsigned>(3)), 4);
  EXPECT_EQ(ceil_round_up(static_cast<size_t>(10), static_cast<size_t>(3)), 4);

  // Note: We cannot test static_assert failure at runtime,
  // as it would cause a compilation error. But we verify that integral types work correctly.
}

// Test round_up_to_multiple_of function
TEST(MathUtilsTest, RoundUpToMultipleOf) {
  // Already aligned cases
  EXPECT_EQ(round_up_to_multiple_of<size_t>(0, 64), 0);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(64, 64), 64);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(128, 64), 128);

  // Cases requiring rounding up
  EXPECT_EQ(round_up_to_multiple_of<size_t>(1, 64), 64);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(65, 64), 128);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(100, 64), 128);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(127, 64), 128);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(129, 64), 192);

  // Different multiples
  EXPECT_EQ(round_up_to_multiple_of<size_t>(10, 3), 12);  // 3 * ceil(10/3) = 3 * 4 = 12
  EXPECT_EQ(round_up_to_multiple_of<size_t>(15, 7), 21);  // 7 * ceil(15/7) = 7 * 3 = 21
}

// Test round_up_to_multiple_of with invalid multiple_of (zero)
TEST(MathUtilsTest, RoundUpToMultipleOf_InvalidMultiple) {
  // Test zero multiple_of (should throw via ceil_round_up)
  EXPECT_THROW(round_up_to_multiple_of<size_t>(10, 0), std::invalid_argument);
}

// Test large value edge cases
TEST(MathUtilsTest, LargeValues) {
  const size_t large_val = (1ULL << 30) + 1;  // NOLINT

  EXPECT_EQ(floor_log2(large_val), 30);
  EXPECT_EQ(ceil_log2(large_val), 31);
  EXPECT_EQ(round_up_to_multiple_of<size_t>(large_val, 64),
            ((large_val / 64) + (large_val % 64 != 0)) * 64);
}

// Test boundary cases for ceil_round_up with maximum values
TEST(MathUtilsTest, CeilRoundUpBoundaryCases) {
  // Test with maximum size_t value
  const size_t max_val = std::numeric_limits<size_t>::max();  // NOLINT
  EXPECT_EQ(ceil_round_up(max_val, max_val), 1);
  EXPECT_EQ(ceil_round_up(max_val, static_cast<size_t>(1)), max_val);

  // Test with divisor = 1
  EXPECT_EQ(ceil_round_up(static_cast<size_t>(100), static_cast<size_t>(1)), 100);
}

}  // namespace alaya
