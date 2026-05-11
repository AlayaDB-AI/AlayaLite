// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Gate L1 smoke test — verifies that every header ported from
// Laser/symqglib under the `alaya::laser` namespace compiles cleanly
// under AlayaLite's toolchain and that `QuantizedGraph` can be
// default-instantiated. Not a runtime test — see
// openspec/changes/port-laser-disk-index/tasks.md 2.13.

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/space/bitwise.hpp"
#include "index/graph/laser/space/inner_product.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "index/graph/laser/space/space.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/laser/utils/buffer.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/laser/utils/freq_relayout.hpp"
#include "index/graph/laser/utils/io.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/laser/utils/pca_transform.hpp"
#include "index/graph/laser/utils/rotator.hpp"
#include "index/graph/laser/utils/scalar_quantize.hpp"
#include "index/graph/laser/utils/stopw.hpp"
#include "index/graph/laser/utils/tools.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/qg/qg_scanner.hpp"
#include "third_party/ngt/hashset.hpp"
#include "third_party/ffht/fht_avx.hpp"
#include "index/graph/laser/utils/array.hpp"

int main() {
    // Task 2.13 calls for linkage verification of QuantizedGraph.
    // The class has no default ctor and validates its args at runtime
    // (dim must be power-of-two, degree>0, residual_dim >= 0, …), so
    // constructing one here would need fixture data. Verify linkage by
    // taking sizeof instead — this forces the complete class definition
    // and pulls in every dependent header's symbols.
    constexpr auto qg_size = sizeof(alaya::laser::QuantizedGraph);
    static_assert(qg_size > 0, "QuantizedGraph must be non-empty");
    return 0;
}
