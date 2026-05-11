# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

import os
import tempfile
from pathlib import Path

import numpy as np
from alayalite import Client, Index

try:
    from alayalite import laser
except ImportError:  # pragma: no cover - exercised on non-LASER builds
    laser = None

from ..base.module import BaseANN


def _laser_metric(metric: str) -> str:
    normalized = str(metric).lower()
    if normalized not in {"l2", "euclidean"}:
        raise ValueError(f"disk_laser adapter supports L2 only, got {metric!r}")
    return "l2"


def _laser_main_dimension(raw_dim: int, override) -> int:
    """Resolve the QG ``main_dimension``.

    main_dim must be a power of two >= 128, <= raw_dim.
    Default is 256 when raw_dim >= 256, else the largest power of two <= raw_dim.
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
    return laser is not None


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

    Uses the unified ``laser.Index.fit()`` entrypoint which mirrors
    ``Laser/reproduce/main.py`` internally: PCA rotation, medoid generation,
    Vamana graph construction, and QG build are all driven by a single call.

    ann-benchmarks provides search-time resource knobs in ``set_query_arguments``.
    The adapter therefore builds with ``auto_load=False`` and loads the built
    prefix later via ``Index.from_prefix(...)``.
    """

    def __init__(self, metric, dim, method_param):
        self.metric = _laser_metric(metric)
        self.raw_dim = int(dim)
        if self.raw_dim < 128:
            raise ValueError(f"disk_laser adapter requires raw dim >= 128, got {self.raw_dim}")
        override = method_param.get("main_dim", method_param.get("pca_dim"))
        self.main_dim = _laser_main_dimension(self.raw_dim, override)

        self.fit_threads = int(method_param.get("fit_threads", 1))
        self.search_threads = int(method_param.get("search_threads", 1))
        self.R = int(method_param.get("R", 64))
        self.L = int(method_param.get("L", 200))
        self.alpha = float(method_param.get("alpha", 1.2))
        self.seed = int(method_param.get("seed", 42))
        self.ep_num = int(method_param.get("ep_num", 300))
        self.beam_width = int(method_param.get("beam_width", 16))
        self.build_ef = int(method_param.get("build_ef", 200))
        self.build_dram_budget_gb = float(method_param.get("build_dram_budget_gb", 1.0))
        self.search_dram_budget_gb = float(method_param.get("search_dram_budget_gb", 2.0))

        ext = method_param.get("external_vamana_path")
        self.external_vamana_path = str(ext) if ext else None
        self.work_root = Path(method_param.get("work_root", "alaya_disk_laser_indices"))
        self.work_root.mkdir(parents=True, exist_ok=True)

        self._built_prefix = None
        self.laser_index = None
        self.res = None
        self.ef = 100

    def fit(self, X: np.array) -> None:
        if not _laser_runtime_supported():
            raise RuntimeError("disk_laser adapter requires a LASER-enabled build")

        vectors = np.ascontiguousarray(X, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"disk_laser adapter expects a 2D matrix, got ndim={vectors.ndim}")
        if vectors.shape[1] != self.raw_dim:
            raise ValueError(f"disk_laser adapter dim mismatch: expected {self.raw_dim}, got {vectors.shape[1]}")
        if vectors.shape[0] < self.raw_dim:
            raise ValueError(
                f"disk_laser adapter needs n_samples ({vectors.shape[0]}) >= raw_dim ({self.raw_dim}) for PCA"
            )

        if self.external_vamana_path is not None:
            raise ValueError(
                "disk_laser adapter: external_vamana_path is not supported with the unified Index.fit() API"
            )

        work_dir = Path(tempfile.mkdtemp(prefix="annbenchmark_disk_laser_", dir=self.work_root))
        name = "dsqg_seg_00000001"

        laser.Index.fit(
            vectors,
            output_dir=work_dir,
            name=name,
            build_params=laser.BuildParams(
                metric=self.metric,
                main_dim=self.main_dim,
                R=self.R,
                L=self.L,
                alpha=self.alpha,
                ef_indexing=self.build_ef,
                ep_num=self.ep_num,
            ),
            seed=self.seed,
            num_threads=self.fit_threads,
            dram_budget_gb=self.build_dram_budget_gb,
            skip_existing=False,
            auto_load=False,
        )
        self._built_prefix = str(work_dir / name)
        self.laser_index = None

    def set_query_arguments(self, ef):
        self.ef = int(ef)
        if self._built_prefix is not None and self.laser_index is None:
            self.laser_index = laser.Index.from_prefix(self._built_prefix, dram_budget_gb=self.search_dram_budget_gb)
        if self.laser_index is not None:
            self.laser_index.set_params(self.ef, self.search_threads, self.beam_width)

    def _check_fit(self) -> None:
        if self.laser_index is None:
            raise RuntimeError("disk_laser adapter: fit() must be called before queries")

    def prepare_query(self, q: np.array, n: int):
        self._check_fit()
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.shape != (self.raw_dim,):
            raise ValueError(f"disk_laser adapter: query shape {q.shape} != ({self.raw_dim},)")
        self.q = q
        self.n = int(n)

    def run_prepared_query(self):
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
