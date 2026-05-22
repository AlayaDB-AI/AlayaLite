# AlayaLite Client User Guide

This guide is based on [`python/src/alayalite/client.py`](https://github.com/AlayaDB-AI/AlayaLite/blob/main/python/src/alayalite/client.py), the main Python entry point for managing AlayaLite objects.

`Client` manages two kinds of objects:

- `Index`: stores vectors only. Use it for pure ANN search, algorithm experiments, and benchmark-style workflows.
- `Collection`: stores `id + document + embedding + metadata`. Use it for RAG, knowledge bases, metadata filtering, and hybrid search.

## Installation

```bash
pip install alayalite
```

```python
import numpy as np
from alayalite import Client
```

## Create A Client

Without `url`, the client only manages objects in the current process and cannot save them through `client.save_*`.

```python
client = Client()
```

With `url`, the client saves to and loads from the given directory.

```python
client = Client(url="./alaya_data")
```

Common management APIs:

```python
client.list_collections()
client.list_indices()

collection = client.get_collection("docs")
index = client.get_index("ann")

collection = client.get_or_create_collection("docs")
index = client.get_or_create_index("ann")
```

Collection names and index names share the same namespace inside one `Client`, so they cannot duplicate each other.

## Pure ANN Search

Use `Index` when you only need nearest-neighbor vector IDs and do not need documents or metadata.

```python
import numpy as np
from alayalite import Client

client = Client()
index = client.create_index(
    name="ann_demo",
    index_type="hnsw",
    metric="l2",
    quantization_type="none",
    capacity=10_000,
)

vectors = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ],
    dtype=np.float32,
)

index.fit(vectors, ef_construction=100, num_threads=1)

query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ids = index.search(query, topk=2, ef_search=50)
print(ids)
```

The returned IDs are internal vector IDs. Vectors passed to the first `fit` call are assigned IDs from `0`; later `insert` calls keep allocating new IDs.

```python
new_id = index.insert(np.array([0.8, 0.2, 0.0], dtype=np.float32), ef=100)
print(new_id)

vec = index.get_data_by_id(new_id)
index.remove(new_id)
```

Batch search:

```python
queries = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

ids = index.batch_search(queries, topk=2, ef_search=50, num_threads=2)
ids, distances = index.batch_search_with_distance(queries, topk=2, ef_search=50, num_threads=2)
```

## LASER On-Disk ANN

LASER is AlayaLite's on-disk Quantized Graph index for large-scale ANN search. It is not created through `Client.create_index(...)`; use the dedicated `alayalite.laser` API instead. For full build and tuning details, see [LASER.md](https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/LASER.md).

```python
from alayalite.laser import BuildParams, Index
```

Use LASER when you need large-scale vector ID retrieval. If you need documents, metadata, or hybrid search, use `Collection`.

### Prerequisites

Supported platforms:

- Linux x86_64: uses the `libaio` backend by default.
- macOS arm64 and x86_64: uses the portable thread-pool backend by default.
- Windows x64 with MSVC 2022: uses the IOCP (I/O Completion Ports) backend by default.
- Linux ARM is not supported yet.

Build options:

- `ALAYA_ENABLE_LASER` controls whether LASER is built.
- It is ON by default on Linux x86_64, macOS, and Windows x64, and OFF elsewhere.
- On Linux, install `libaio` or use the thread-pool fallback.

Platform dependencies:

```bash
# Debian / Ubuntu
sudo apt-get install libaio-dev

# Fedora / RHEL
sudo dnf install libaio-devel

# macOS (Homebrew)
brew install libomp

# Windows
# Install Visual Studio 2022 with the Desktop development with C++ workload.
# MSVC ships the OpenMP runtime used by LASER; no extra package is required.
```

For a package install, add the LASER runtime dependencies to the same environment:

```bash
pip install scikit-learn faiss-cpu tqdm psutil matplotlib tomli
```

When working from a cloned repository, install the developer dependency group instead:

```bash
uv sync --group laser
```

Common CMake configurations:

```bash
# Linux x86_64, default libaio backend
cmake -B build/Release -DALAYA_ENABLE_LASER=ON

# macOS default
cmake -B build/Release -DALAYA_ENABLE_LASER=ON

# Windows x64 default, run from a VS 2022 developer shell
cmake -B build/Release -G "Visual Studio 17 2022" -A x64 -DALAYA_ENABLE_LASER=ON

# Linux fallback without libaio
cmake -B build/Release -DALAYA_ENABLE_LASER=ON -DALAYA_LASER_USE_THREADPOOL=ON

# Disable LASER
cmake -B build/Release -DALAYA_ENABLE_LASER=OFF
```

Data and parameter constraints:

- Input vectors must be `float32`.
- Raw vector dimension must be `raw_dim >= 128`.
- LASER currently supports `metric="l2"` or `"euclidean"`; both are normalized to `l2`.
- `main_dim` is the PCA target dimension. If omitted, PCA is skipped.
- If `main_dim` is set, it must be a power of two and satisfy `64 <= main_dim <= raw_dim`.
- Query vector dimension must match the raw dimension, not `main_dim`.

### Build From A Numpy Array

```python
import numpy as np
from alayalite.laser import BuildParams, Index

rng = np.random.default_rng(42)
vectors = rng.normal(size=(10000, 128)).astype(np.float32)
queries = rng.normal(size=(5, 128)).astype(np.float32)

idx = Index.fit(
    vectors,
    output_dir="./laser_build",
    name="demo",
    build_params=BuildParams(
        metric="l2",
        main_dim=128,
        R=64,
        L=200,
        alpha=1.2,
        ef_indexing=200,
        ep_num=300,
    ),
    seed=42,
    num_threads=8,
    dram_budget_gb=1.0,
    auto_load=True,
)

idx.set_params(ef_search=200, num_threads=1, beam_width=16)

one_result = idx.search(queries[0], k=10)
batch_result = idx.batch_search(queries, k=10)
```

`search` and `batch_search` return internal vector IDs. LASER does not store documents or metadata.

### Build From `.fbin`

LASER commonly consumes DiskANN `.fbin` files: `<int32 N><int32 dim>` followed by `N * dim` row-major `float32` values.

```python
from alayalite.laser import BuildParams, Index

idx = Index.fit(
    "/path/to/base.fbin",
    output_dir="/path/to/build/laser",
    name="dsqg_gist",
    build_params=BuildParams(
        metric="l2",
        main_dim=256,
        R=64,
        L=200,
        alpha=1.2,
        ef_indexing=200,
        ep_num=300,
    ),
    seed=42,
    num_threads=48,
    dram_budget_gb=32.0,
)
```

Build artifacts are written under `output_dir` with `name` as the prefix:

- `<prefix>_pca_base.fbin`
- `<prefix>_pca.bin`, only when PCA is used.
- `<prefix>_medoids` and `<prefix>_medoids_indices`
- `<prefix>_vamana_graph.index`
- `<prefix>_R<R>_MD<main_dim>.index`
- `<prefix>_seed.txt`

### Reuse Build Artifacts

`seed` is the master seed for randomized LASER build steps such as PCA sampling, medoids, Vamana, and rotator generation. By default, `skip_existing=True`; existing artifacts are reused when their headers and `<prefix>_seed.txt` match the requested build.

```python
idx = Index.fit(
    vectors,
    output_dir="./laser_build",
    name="demo",
    seed=42,
    skip_existing=True,
    auto_load=False,  # build only, do not load for search immediately
)
```

For more stable local rebuilds, use `num_threads=1` and run with single-threaded BLAS/OpenMP settings.

### Load An Existing LASER Index

When build and search happen in different processes, load by prefix:

```python
from alayalite.laser import Index

idx = Index.from_prefix("/path/to/build/laser/dsqg_gist", dram_budget_gb=1.0)
idx.set_params(ef_search=200, num_threads=1, beam_width=16)

ids = idx.batch_search(queries, k=10)
```

The prefix does not include `_R64_MD256.index`. For this file:

```text
/path/to/build/laser/dsqg_gist_R64_MD256.index
```

load with:

```python
Index.from_prefix("/path/to/build/laser/dsqg_gist")
```

### Search Parameters

```python
idx.set_params(
    ef_search=200,
    num_threads=1,
    beam_width=16,
)
```

- `ef_search`: search candidate size. Larger values usually improve recall and reduce speed.
- `num_threads`: search thread count; `0` means using available CPU cores.
- `beam_width`: disk-search beam width, affecting I/O parallelism and recall/speed tradeoffs.

### CLI

[`examples/laser/main.py`](https://github.com/AlayaDB-AI/AlayaLite/blob/main/examples/laser/main.py) provides a TOML-driven build/search CLI:

```bash
# Build and search
uv run examples/laser/main.py -c examples/laser/configs/gist.toml all

# Build only
uv run examples/laser/main.py -c examples/laser/configs/gist.toml build

# Search an existing build
uv run examples/laser/main.py -c examples/laser/configs/gist.toml search \
    --threads 1 --efs 100 200 300
```

Minimal TOML shape:

```toml
seed = 42

[dataset]
name = "gist"
metric = "l2"
degree = 64
main_dimension = 256

[paths]
base = "/path/to/base.fbin"
query = "/path/to/query.fbin"
gt = "/path/to/gt.ibin"
output = "/path/to/build/laser"

[build]
build_threads = 48
ef_indexing = 200

[build_vamana]
L = 200
alpha = 1.2
dram_budget_gb = 32.0

[search]
topk = 10
threads = 1
beam_width = 16
dram_budget = 1.0
ep_num = 300
efs = [100, 200, 300]
```

### LASER And DiskCollection

`DiskCollection(index_type="disk_laser")` can import precomputed LASER artifacts as disk collection segments. This is a lower-level disk collection workflow, not the regular `Client.create_collection(...)` document/metadata collection path.

```python
import numpy as np
from alayalite import DiskCollection, MetricType

col = DiskCollection(
    path="./disk_laser_collection",
    dim=128,
    metric=MetricType.L2,
    index_type="disk_laser",
)

labels = np.array(range(10000), dtype=np.uint64)
col.import_laser_segment("./laser_segment_dir", labels)

query = np.random.default_rng(7).normal(size=128).astype(np.float32)
hits = col.search(query, k=5, ef=128, beam_width=4)
print(hits)
```

`labels` must be a 1D C-contiguous `numpy.uint64` array whose length matches the LASER index row count. `disk_laser` availability depends on whether the current build enabled LASER.

## Collection With Documents And Metadata

Use `Collection` when search results should return business IDs, documents, and metadata.

```python
client = Client()
collection = client.create_collection(
    name="docs",
    metric="l2",
    quantization_type="none",
    capacity=10_000,
)

items = [
    (
        "doc-1",
        "AlayaLite supports vector search.",
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        {"category": "database", "year": 2025},
    ),
    (
        "doc-2",
        "Hybrid search combines vector search and metadata filters.",
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
        {"category": "search", "year": 2025},
    ),
    (
        "doc-3",
        "This document is about cooking.",
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        {"category": "life", "year": 2024},
    ),
]

collection.insert(items)
```

Use `batch_query` for vector search with document payloads:

```python
result = collection.batch_query(
    vectors=[[1.0, 0.0, 0.0]],
    limit=2,
    ef_search=50,
    num_threads=1,
)

print(result["id"][0])
print(result["document"][0])
print(result["metadata"][0])
print(result["distance"][0])
```

Return values are grouped by query:

```python
{
    "id": [["doc-1", "doc-2"]],
    "document": [["...", "..."]],
    "metadata": [[{...}, {...}]],
    "distance": [[0.0, 0.02]],
}
```

`ef_search` must be greater than or equal to `limit`.

## Hybrid Search

Hybrid search combines vector ANN search with metadata filtering. Use `Collection.hybrid_query`; for better filtering performance, put frequently filtered fields in `indexed_fields` when creating the collection.

```python
collection = client.create_collection(
    name="hybrid_docs",
    metric="l2",
    quantization_type="none",
    capacity=10_000,
    indexed_fields=["category", "year"],
)

collection.insert(items)
```

Search for vectors similar to the query and matching `category == "search"`:

```python
result = collection.hybrid_query(
    vectors=[[1.0, 0.0, 0.0]],
    limit=2,
    metadata_filter={"category": "search"},
    ef_search=50,
)

print(result["id"][0])
```

`hybrid_query` currently returns business IDs only. Fetch documents and metadata with `get_by_id`:

```python
ids = result["id"][0]
docs = collection.get_by_id(ids)
print(docs["document"])
```

Batch hybrid search:

```python
result = collection.hybrid_query(
    vectors=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    limit=2,
    metadata_filter={"year": {"$ge": 2025}},
    ef_search=50,
    num_threads=2,
)
```

## Metadata Filter Syntax

Equality:

```python
{"category": "search"}
```

Comparisons:

```python
{"year": {"$gt": 2024}}
{"year": {"$ge": 2025}}
{"year": {"$lt": 2026}}
{"year": {"$le": 2025}}
{"category": {"$eq": "database"}}
```

Set membership:

```python
{"category": {"$in": ["database", "search"]}}
```

Logical combinations:

```python
{
    "$and": [
        {"year": {"$ge": 2025}},
        {"category": {"$in": ["database", "search"]}},
    ]
}
```

```python
{
    "$or": [
        {"category": "database"},
        {"category": "search"},
    ]
}
```

Metadata-only filtering without vector search:

```python
result = collection.filter_query(
    metadata_filter={"category": {"$in": ["database", "search"]}},
    limit=10,
)

print(result["id"])
print(result["document"])
```

## Hybrid Execution Hints

`hybrid_query` supports `filter_execution_hint`:

- `None` or `"auto"`: choose automatically.
- `"disable"`: disable filter acceleration.
- `"bitset_prefilter"`: build a metadata bitset before vector search.
- `"iterative_filter"`: filter while searching.

```python
result = collection.hybrid_query(
    vectors=[[1.0, 0.0, 0.0]],
    limit=2,
    metadata_filter={"year": {"$ge": 2025}},
    ef_search=50,
    filter_execution_hint="iterative_filter",
)
```

Use the default unless you have a specific reason to tune the strategy.

## Updates And Deletes

Get records by business ID:

```python
result = collection.get_by_id(["doc-1", "doc-2"])
print(result["id"])
print(result["document"])
print(result["metadata"])
```

Insert or update records:

```python
collection.upsert(
    [
        (
            "doc-1",
            "Updated document text.",
            np.array([0.95, 0.05, 0.0], dtype=np.float32),
            {"category": "database", "year": 2026},
        )
    ]
)
```

Delete by ID:

```python
collection.delete_by_id(["doc-3"])
```

Delete by metadata filter:

```python
deleted_count = collection.delete_by_filter({"year": {"$lt": 2025}})
print(deleted_count)
```

After many deletes, rebuild the graph index to reduce the effect of deleted records on recall and performance:

```python
collection.reindex(ef_construction=400, num_threads=4)
```

## Persistence

Objects created through `Client(url=...)` can be saved to disk.

```python
client = Client(url="./alaya_data")

collection = client.create_collection("docs")
collection.insert(items)
client.save_collection("docs")

index = client.create_index("ann")
index.fit(vectors, ef_construction=100, num_threads=1)
client.save_index("ann")
```

Reopen the same directory to load existing objects:

```python
client = Client(url="./alaya_data")

collection = client.get_collection("docs")
index = client.get_index("ann")
```

Delete from memory:

```python
client.delete_collection("docs")
client.delete_index("ann")
```

Delete from memory and disk:

```python
client.delete_collection("docs", delete_on_disk=True)
client.delete_index("ann", delete_on_disk=True)
```

Reset all objects managed by the current client:

```python
client.reset()
```

Delete matching on-disk object directories as well:

```python
client.reset(delete_on_disk=True)
```

## Common Index Parameters

These parameters can be passed to `create_index` or `create_collection`:

```python
client.create_collection(
    name="docs",
    index_type="hnsw",
    metric="cosine",
    quantization_type="sq8",
    data_type=np.float32,
    id_type=np.uint32,
    capacity=1_000_000,
    max_nbrs=32,
    build_threads=8,
    indexed_fields=["category", "year"],
)
```

Common values:

- `index_type`: `"hnsw"`, `"nsg"`, `"fusion"`.
- `metric`: `"l2"`, `"euclidean"`, `"ip"`, `"cosine"`, `"cos"`.
- `quantization_type`: `"none"`, `"sq8"`, `"sq4"`, `"rabitq"`.
- `data_type`: `np.float32`, `np.int8`, `np.uint8`.
- `id_type`: `np.uint32`, `np.uint64`.
- `capacity`: index capacity; inserting beyond it raises an error.
- `indexed_fields`: metadata fields used to accelerate filtering and hybrid search.

For `cosine` or `cos`, AlayaLite normalizes inserted and queried vectors.

## Choosing Index Or Collection

Use `Index` when you only need nearest-neighbor vector IDs:

```python
index = client.create_index("ann")
index.fit(vectors, ef_construction=100, num_threads=4)
ids = index.search(query, topk=10, ef_search=100)
```

Use `Collection` when you need documents, metadata, or hybrid search:

```python
collection = client.create_collection("docs", indexed_fields=["category"])
collection.insert(items)
result = collection.hybrid_query(
    vectors=[query],
    limit=10,
    metadata_filter={"category": "database"},
    ef_search=100,
)
```

## Notes

- `Index.fit` can be called only once. Use `index.insert` to append vectors.
- `Collection` initializes its underlying index on the first `insert`; queries before that will fail.
- Each `Collection.insert` item is `(item_id, document, embedding, metadata)`.
- Returned `item_id` values are usually strings. If you insert integer IDs, query results may return `"1"`.
- Query vector dimensions for `batch_query` and `hybrid_query` must match the indexed vector dimension.
- `batch_query` requires `ef_search >= limit`.
- `client.save_collection` and `client.save_index` require creating the client with `url`.
