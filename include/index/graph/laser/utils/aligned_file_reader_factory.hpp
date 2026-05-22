// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <memory>

#include "index/graph/laser/utils/aligned_file_reader.hpp"

#if defined(ALAYA_LASER_USE_THREADPOOL)
  #include "index/graph/laser/utils/threadpool_file_reader.hpp"
#elif defined(ALAYA_LASER_USE_IOCP)
  #include "index/graph/laser/utils/iocp_file_reader.hpp"
#endif

inline std::unique_ptr<AlignedFileReader> make_aligned_file_reader() {
#if defined(ALAYA_LASER_USE_THREADPOOL)
  return std::make_unique<ThreadPoolFileReader>();
#elif defined(ALAYA_LASER_USE_IOCP)
  return std::make_unique<IOCPFileReader>();
#else
  return std::make_unique<LinuxAlignedFileReader>();
#endif
}
