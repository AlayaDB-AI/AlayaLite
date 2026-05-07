# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from pathlib import Path

import numpy as np
from alayalite import Client, Index

try:
    from alayalite import laser
except ImportError:  # pragma: no cover - exercised on non-LASER builds
    laser = None

try:
    from alayalite import vamana
except ImportError:  # pragma: no cover - exercised on non-LASER builds
    vamana = None

try:  # sklearn-backed PCA from alayalite.laser.pca; required for disk_laser path.
    from alayalite.laser.pca import fit_incremental_pca, save_pca_params
except ImportError:  # pragma: no cover
    fit_incremental_pca = None
    save_pca_params = None

try:  # faiss-backed kmeans medoid generation, mirroring native step_medoid.
    from alayalite.laser.medoid import generate_and_save_medoids
except ImportError:  # pragma: no cover
    generate_and_save_medoids = None

from ..base.module import BaseANN


def _laser_metric(metric: str) -> str:
    normalized = str(metric).lower()
    if normalized not in {"l2", "euclidean"}:
        raise ValueError(f"disk_laser adapter supports L2 only, got {metric!r}")
    return "l2"


def _laser_main_dimension(raw_dim: int, override) -> int:
    """Resolve the QG ``main_dimension``.

    Mirrors upstream Laser's gist/cohere ``main_dimension = 256`` config
    (`/md1/huangliang/alaya-dev/Laser/reproduce/configs/gist.toml`). The QG
    splits a PCA-rotated input vector into two parts: the leading
    ``main_dim`` axes are 1-bit RaBitQ-quantized for fast pruning, the
    trailing ``raw_dim - main_dim`` axes are stored as fp32 residual for
    re-ranking. Without that residual the small-fixture recall collapses
    (Codex's gist-960 fixture, see session notes).

    main_dim must be a power of two ≥ 128, ≤ raw_dim. Default is 256 when
    raw_dim ≥ 256, else the largest power of two ≤ raw_dim.
    """
    if override is not None:
        target = int(override)
    else:
        target = 256 if raw_dim >= 256 else 1 << (int(raw_dim).bit_length() - 1)

    if target <= 0 or (target & (target - 1)) != 0:
        raise ValueError(f"main_dim must be a power of two, got {target}")
    if target < 128:
        raise ValueError(f"main_dim must be >= 128 (LASER QG floor), got {target}")
    if target > raw_dim:
        raise ValueError(f"main_dim ({target}) must be <= raw dim ({raw_dim})")
    return target


def _laser_runtime_supported() -> bool:
    """LASER runtime probe: native bindings + sklearn-backed PCA all importable.

    The DiskCollection-based probe was removed: this adapter no longer goes
    through ``DiskCollection.import_laser_segment`` (whose v1 simplification
    forces ``main_dim == dim`` and forfeits the QG residual). We now drive
    ``alayalite.laser.Index`` directly the way ``Laser/reproduce/main.py``
    does, so the only supported-build signal we need is that the imports
    above succeeded (they raise ImportError on non-LASER wheels).
    """
    return (
        laser is not None
        and vamana is not None
        and fit_incremental_pca is not None
        and save_pca_params is not None
        and generate_and_save_medoids is not None
    )


def _write_fbin(path: Path, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    with path.open("wb") as f:
        np.array([vectors.shape[0], vectors.shape[1]], dtype=np.int32).tofile(f)
        vectors.tofile(f)


class AlayaLite(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.index_save_dir = "alaya_indices"
        self.client = Client(self.index_save_dir)
        self.index = None
        self.ef = None
        self.dim = dim
        self.metric = metric

        self.index_type = method_param["index_type"]
        self.quantization_type = method_param["quantization_type"]
        self.fit_threads = method_param["fit_threads"]
        self.search_threads = method_param["search_threads"]
        self.R = method_param["R"]
        self.L = method_param["L"]
        self.M = method_param["M"]

        self.save_index_name = f"alayalite_index_it_{self.index_type}_qt_{self.quantization_type}_dim_{self.dim}_metric_{self.metric}_M{self.M}.idx"
        print("alaya init done")

    def fit(self, X: np.array) -> None:
        if os.path.exists(os.path.join(self.index_save_dir, self.save_index_name)):
            self.index = Index.load(self.index_save_dir, self.save_index_name)
            print("load index from cache")
        else:
            X = X.astype(np.float32)
            self.index = self.client.create_index(
                name=self.save_index_name,
                metric=self.metric,
                quantization_type=self.quantization_type,
                capacity=X.shape[0],
            )
            self.index.fit(vectors=X, num_threads=self.fit_threads)
            self.client.save_index(self.save_index_name)
            print("save index to cache")

    def set_query_arguments(self, ef):
        self.ef = int(ef)

    def prepare_query(self, q: np.array, n: int):
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.index.search(query=self.q, topk=self.n, ef_search=self.ef)

    def batch_query(self, X: np.array, n: int) -> None:
        self.res = self.index.batch_search(queries=X, topk=n, ef_search=self.ef)

    def get_prepared_query_results(self):
        return self.res

    def get_batch_results(self) -> np.array:
        return self.res

    def __str__(self) -> str:
        return "AlayaLite"


class AlayaLiteDiskLaser(BaseANN):
    """ann-benchmarks adapter for the LASER on-disk QG index.

    Architecture mirrors `examples/laser/main.py` exactly:

      1. PCA rotates the base vectors (n_components = raw_dim, rotation
         only, no truncation). Variance is concentrated in the leading axes.
      2. Vamana builds a graph on the **raw** fp32 base — see
         `examples/laser/main.py:326`. Earlier adapter revisions fed the
         PCA-rotated base into Vamana and produced a structurally
         different graph; the current path matches native byte-for-byte.
      3. ``alayalite.laser.Index(main_dimension=M, dimension=D)`` builds
         the disk QG with ``main_dim < dim`` — the leading M axes are
         1-bit RaBitQ-quantized, the trailing ``D - M`` axes are kept as
         fp32 residual for re-ranking. The QG reads
         ``<prefix>_pca_base.fbin`` for the rotated input
         (``qg_builder.hpp:123``).
      4. ``Index.load`` opens the file reader; queries are PCA-rotated and
         fed through ``Index.search`` / ``Index.batch_search`` directly.

    DiskCollection is *not* used. Its v1 importer hardcodes
    ``main_dim == dim``, which forfeits the QG residual and collapses
    recall on small fixtures (gist-960 with 2048 vectors → 0.3% recall
    in earlier sessions). The standalone Laser pipeline does not have
    this constraint and routinely hits high recall with main_dim=256.
    """

    def __init__(self, metric, dim, method_param):
        self.metric = _laser_metric(metric)
        self.raw_dim = int(dim)
        if self.raw_dim < 128:
            raise ValueError(f"disk_laser adapter requires raw dim >= 128, got {self.raw_dim}")
        # Accept legacy `pca_dim` alias for back-compat with earlier sessions.
        override = method_param.get("main_dim", method_param.get("pca_dim"))
        self.main_dim = _laser_main_dimension(self.raw_dim, override)

        self.fit_threads = int(method_param.get("fit_threads", 1))
        self.search_threads = int(method_param.get("search_threads", 1))
        self.R = int(method_param.get("R", 64))
        self.L = int(method_param.get("L", 100))
        self.alpha = float(method_param.get("alpha", 1.2))
        self.seed = int(method_param.get("seed", 42))  # rotator_seed
        # Per-stage RNG seeds mirroring native gist1m_seed_*.toml. All
        # default to 42 so a method_param without overrides reproduces the
        # seeded native run end-to-end. Native fields:
        #   pca_seed = 42      (PCA training-sample selection)
        #   medoid_seed = 42   (faiss kmeans for medoid generation)
        #   rotator_seed = 42  (LASER QG internal RaBitQ rotator — `seed`)
        self.pca_seed = int(method_param.get("pca_seed", self.seed))
        self.medoid_seed = int(method_param.get("medoid_seed", self.seed))
        # native `sample_vectors_from_fbin(.. sample_ratio=0.25)` default,
        # `[search].ep_num = 300` default for 300-medoid kmeans.
        self.pca_sample_ratio = float(method_param.get("pca_sample_ratio", 0.25))
        self.ep_num = int(method_param.get("ep_num", 300))
        # Default beam_width=16 mirrors `Laser/reproduce/main.py:39`.
        self.beam_width = int(method_param.get("beam_width", 16))
        self.build_ef = int(method_param.get("build_ef", 200))
        self.build_num_iter = int(method_param.get("build_num_iter", 3))
        # Defaults mirror Laser/reproduce/configs/gist1m_seed_alayaV_origP.toml:
        # [build_vamana].dram_budget_gb = 1.0, [search].dram_budget = 2.0.
        self.build_dram_budget_gb = float(method_param.get("build_dram_budget_gb", 1.0))
        self.search_dram_budget_gb = float(method_param.get("search_dram_budget_gb", 2.0))
        # Optional pre-built Vamana graph for byte-reproducible runs against a
        # frozen native reference. When set, fit() skips Vamana build entirely
        # and feeds this path directly to laser.Index.build_index. The
        # `budget_estimator` formula was unified in commit 76b6475 (2026-04-23),
        # so a fresh build today picks a different shard count than pre-4-23
        # native reference graphs — pinning the input graph removes that
        # variable from cross-version comparisons.
        ext = method_param.get("external_vamana_path")
        self.external_vamana_path = str(ext) if ext else None
        self.work_root = Path(method_param.get("work_root", "alaya_disk_laser_indices"))
        self.work_root.mkdir(parents=True, exist_ok=True)

        # `self.pca` is only kept around for the legacy `query before fit`
        # sentinel; LASER applies the rotation internally now via the saved
        # `<prefix>_pca.bin` (mirrors native step_search). Adapter no longer
        # rotates queries externally in prepare_query.
        self.pca = None
        self.laser_index = None
        self.res = None
        self.ef = 100

    def fit(self, X: np.array) -> None:
        if not _laser_runtime_supported():
            raise RuntimeError(
                "disk_laser adapter requires a LASER-enabled Linux build with sklearn-backed alayalite.laser.pca"
            )

        vectors = np.ascontiguousarray(X, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"disk_laser adapter expects a 2D matrix, got ndim={vectors.ndim}")
        if vectors.shape[1] != self.raw_dim:
            raise ValueError(f"disk_laser adapter dim mismatch: expected {self.raw_dim}, got {vectors.shape[1]}")
        if vectors.shape[0] < self.raw_dim:
            # IncrementalPCA needs n_samples >= n_components; fail early with
            # a useful message instead of an opaque sklearn error.
            raise ValueError(
                f"disk_laser adapter needs n_samples ({vectors.shape[0]}) >= raw_dim ({self.raw_dim}) "
                "for PCA fitting. For tiny fixtures, lower raw_dim or skip the disk_laser path."
            )

        # Validate external Vamana path BEFORE the expensive PCA pass — fail
        # loudly so a typo doesn't silently cost ~1 minute of PCA work.
        if self.external_vamana_path is not None and not Path(self.external_vamana_path).is_file():
            raise FileNotFoundError(
                f"disk_laser adapter: external_vamana_path does not point to a regular file: "
                f"{self.external_vamana_path}"
            )

        # PCA rotation, n_components=raw_dim. Mirrors native `step_pca`
        # (examples/laser/main.py:386,391):
        #   vectors, sample_vecs = sample_vectors_from_fbin(base, seed=pca_seed)
        #   pca = fit_incremental_pca(sample_vecs, n_components=d)
        # Sampling on the in-memory training matrix uses the same seeded
        # `np.random.default_rng(pca_seed).choice(n, sample_size,
        # replace=False)` recipe as the native fbin sampler so two runs
        # with identical input + seed produce byte-equivalent samples.
        n_samples = vectors.shape[0]
        sample_size = max(self.raw_dim, int(self.pca_sample_ratio * n_samples))
        if sample_size >= n_samples:
            sample_vecs = vectors
        else:
            rng = np.random.default_rng(self.pca_seed)
            sample_indices = np.sort(rng.choice(n_samples, sample_size, replace=False))
            sample_vecs = vectors[sample_indices]

        self.pca = fit_incremental_pca(sample_vecs, n_components=self.raw_dim)
        rotated = np.ascontiguousarray(self.pca.transform(vectors), dtype=np.float32)

        work_dir = Path(tempfile.mkdtemp(prefix="annbenchmark_disk_laser_", dir=self.work_root))
        prefix = work_dir / "dsqg_seg_00000001"
        # `qg_builder.hpp:123` derives the QG input path as
        # `<prefix>_pca_base.fbin`, NOT a free-form filename. Vamana takes
        # a separate raw fbin to mirror native examples/laser/main.py:326,
        # which passes `cfg["base"]` (the raw, un-rotated input) into
        # `vamana.build_index(data_path=...)`. Building Vamana on the
        # PCA-rotated base produces a structurally different graph
        # (smaller .index, lower recall ceiling — Codex 2026-05 root-cause).
        pca_base_path = prefix.with_name(prefix.name + "_pca_base.fbin")
        _write_fbin(pca_base_path, rotated)

        # Persist the trained PCA so LASER's `index.load` picks it up at
        # `<prefix>_pca.bin` and applies the rotation internally during
        # search. Mirrors `examples/laser/main.py:392`. Without this the
        # adapter would have to rotate queries externally (the legacy
        # path that produced the 0.3-0.9pp recall + 14-24% QPS deficit
        # vs native B in the 2026-05-06 audit).
        pca_params_path = prefix.with_name(prefix.name + "_pca.bin")
        save_pca_params(self.pca, str(pca_params_path))

        if self.external_vamana_path is not None:
            # Reuse a pre-built Vamana graph verbatim. Skip raw_base.fbin
            # write and Vamana build — the QG only needs the graph file +
            # `<prefix>_pca_base.fbin` (already written above).
            vamana_path = Path(self.external_vamana_path)
        else:
            raw_base_path = work_dir / "raw_base.fbin"
            vamana_path = work_dir / "graph.index"
            _write_fbin(raw_base_path, vectors)
            vamana.build_index(
                data_path=str(raw_base_path),
                output_path=str(vamana_path),
                R=self.R,
                L=self.L,
                alpha=self.alpha,
                seed=self.seed,
                num_threads=self.fit_threads,
                dram_budget_gb=self.build_dram_budget_gb,
            )

        # Generate `<prefix>_medoids` + `<prefix>_medoids_indices` mirroring
        # native `step_medoid` (examples/laser/main.py:417). LASER uses
        # these as multi-entry-point starting nodes during search; missing
        # them forces the QG to fall back to a single global medoid, which
        # increases beam expansion at low EF and hurts QPS. Native default
        # is `[search].ep_num = 300` clusters seeded with `medoid_seed = 42`.
        medoid_indices_path = prefix.with_name(prefix.name + "_medoids_indices")
        medoid_vectors_path = prefix.with_name(prefix.name + "_medoids")
        generate_and_save_medoids(
            str(pca_base_path),
            str(medoid_indices_path),
            str(medoid_vectors_path),
            self.ep_num,
            seed=self.medoid_seed,
        )

        self.laser_index = laser.Index(
            index_type="QG",
            metric=self.metric,
            num_elements=rotated.shape[0],
            main_dimension=self.main_dim,
            dimension=self.raw_dim,
            degree_bound=self.R,
            rotator_seed=self.seed,
            rotator_dump_path="",
        )
        self.laser_index.build_index(
            str(vamana_path),
            str(prefix),
            EF=self.build_ef,
            num_iter=self.build_num_iter,
            num_thread=self.fit_threads,
        )
        self.laser_index.load(str(prefix), self.search_dram_budget_gb)
        # Initialise default search params so a query before
        # set_query_arguments() doesn't hit an uninitialised QG state.
        self.laser_index.set_params(self.ef, self.search_threads, self.beam_width)

    def set_query_arguments(self, ef):
        self.ef = int(ef)
        if self.laser_index is not None:
            self.laser_index.set_params(self.ef, self.search_threads, self.beam_width)

    def _check_fit(self) -> None:
        # Use ``laser_index`` (the last thing fit() sets) as the
        # "fit() called" sentinel — ``self.pca`` is no longer the
        # canonical signal because LASER applies PCA internally now.
        if self.laser_index is None:
            raise RuntimeError("disk_laser adapter: fit() must be called before queries")

    def prepare_query(self, q: np.array, n: int):
        self._check_fit()
        # Pass the raw query verbatim. LASER's `index.load(<prefix>_pca.bin)`
        # already loaded the PCA matrix; `self.laser_index.search()`
        # applies the rotation internally before quantized lookup.
        # Mirrors native `examples/laser/main.py:572`:
        #     pred = index.search(query[i], topk)   # query[i] is raw 960d
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (self.raw_dim,):
            raise ValueError(f"disk_laser adapter: query shape {q.shape} != ({self.raw_dim},)")
        self.q = q
        self.n = int(n)

    def run_prepared_query(self):
        # laser.Index.search returns numpy uint32 array of k PIDs (0..N-1,
        # the order vectors were written to base.fbin == ann-benchmarks's
        # train order, so no label translation is needed).
        results = self.laser_index.search(self.q, self.n)
        self.res = [int(x) for x in results]

    def batch_query(self, X: np.array, n: int) -> None:
        self._check_fit()
        queries = np.ascontiguousarray(X, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.raw_dim:
            raise ValueError(f"disk_laser adapter: batch query shape {queries.shape} != (?, {self.raw_dim})")
        raw = self.laser_index.batch_search(queries, int(n))
        self.res = [[int(x) for x in row] for row in raw]

    def get_prepared_query_results(self):
        return self.res

    def get_batch_results(self) -> np.array:
        return self.res

    def __str__(self) -> str:
        return "AlayaLiteDiskLaser"
