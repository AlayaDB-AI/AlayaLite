// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>

namespace alaya {

enum class FilterExecHint : uint8_t {
  kAuto = 0,
  kDisableIterative = 1,
  kIterativeFilter = 2,
};

struct SearchInfo {
  uint32_t topk_ = 0;
  uint32_t ef_ = 0;
  FilterExecHint filter_exec_hint_ = FilterExecHint::kAuto;
};

}  // namespace alaya
