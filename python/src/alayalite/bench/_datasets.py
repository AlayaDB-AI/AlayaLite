# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Dataset adapters for the unified DiskCollection benchmark harness."""

# pylint: disable=inconsistent-quotes,missing-class-docstring,cell-var-from-loop
# Notes:
#   - inconsistent-quotes: f-string subscript syntax requires single quotes
#     inside double-quoted f-strings on Python 3.10 (.venv-py310-backup).
#   - missing-class-docstring: DatasetSpec is a frozen dataclass; the field
#     comments / module docstring serve as its documentation.
#   - cell-var-from-loop: the `lambda: f.read(...)` closure inside
#     `_sha16_files` / `_sha16_directory` is invoked synchronously within
#     the same loop iteration's `with` block, so the capture-of-f is safe.

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


class DatasetMissing(RuntimeError):
    """Raised when an optional real benchmark dataset is not present."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    vectors: np.ndarray
    labels: np.ndarray
    queries: np.ndarray
    ground_truth: Optional[np.ndarray]
    sha16: str
    row_count: Optional[int] = None
    dimension: Optional[int] = None

    @property
    def n(self) -> int:
        return int(self.row_count if self.row_count is not None else self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.dimension if self.dimension is not None else self.vectors.shape[1])


def _sha16_arrays(*items) -> str:
    """Digest a sequence of np.ndarray and/or raw bytes into a 16-hex prefix.

    Accepting raw bytes alongside arrays lets callers fold non-array
    discriminators into the hash (e.g. metric string for synth, since the
    same vectors+queries produce different ground truth under L2 vs IP vs
    COS — without folding metric in, two cross-metric runs would share a
    `dataset_sha256_prefix` even though their results are not comparable).
    """
    digest = hashlib.sha256()
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            digest.update(bytes(item))
        else:
            digest.update(np.ascontiguousarray(item).tobytes())
    return digest.hexdigest()[:16]


def _sha16_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _sha16_directory(root: Path) -> str:
    """Digest all regular files under `root` recursively, in sorted order.

    Used to fold a LASER `--laser-src-dir` artifact into the dataset hash
    so two runs with the same vectors/queries but different prebuilt LASER
    indices report distinct `dataset_sha256_prefix` values.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _labels(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.uint64)


def _distance_order(vectors: np.ndarray, query: np.ndarray, metric: str, k: int) -> np.ndarray:
    kth = max(0, min(k - 1, vectors.shape[0] - 1))
    if metric == "IP":
        scores = vectors @ query
        top = np.argpartition(-scores, kth)[:k]
        return top[np.argsort(-scores[top], kind="stable")]
    if metric == "COS":
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = float(np.linalg.norm(query))
        denom = np.maximum(vector_norms * query_norm, np.float32(1e-12))
        scores = (vectors @ query) / denom
        top = np.argpartition(-scores, kth)[:k]
        return top[np.argsort(-scores[top], kind="stable")]

    diffs = vectors - query
    distances = np.einsum("ij,ij->i", diffs, diffs)
    top = np.argpartition(distances, kth)[:k]
    return top[np.argsort(distances[top], kind="stable")]


def exact_ground_truth(
    vectors: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    *,
    metric: str,
    k: int = 100,
) -> np.ndarray:
    if vectors.shape[0] == 0:
        raise ValueError("cannot compute ground truth for an empty dataset")
    depth = min(int(k), vectors.shape[0])
    rows = []
    for query in queries:
        order = _distance_order(vectors, query, metric, depth)
        rows.append(labels[order])
    return np.ascontiguousarray(np.vstack(rows).astype(np.uint64, copy=False))


def load_synth(n: int, dim: int, queries: int, seed: int, *, metric: str = "L2") -> DatasetSpec:
    if n <= 0:
        raise ValueError("--n must be > 0 for synth")
    if dim <= 0:
        raise ValueError("--dim must be > 0 for synth")
    if queries <= 0:
        raise ValueError("--queries must be > 0")

    rng = np.random.default_rng(seed)
    vectors = np.ascontiguousarray(rng.standard_normal((n, dim)).astype(np.float32))
    query_vectors = np.ascontiguousarray(rng.standard_normal((queries, dim)).astype(np.float32))
    labels = _labels(n)
    gt = exact_ground_truth(vectors, labels, query_vectors, metric=metric, k=min(100, n))
    # Fold metric into the hash: same (n, dim, queries, seed) produces
    # different ground truth across L2 / IP / COS, so the hash MUST diverge
    # too; otherwise cross-metric runs share a `dataset_sha256_prefix` and
    # downstream diff tools silently merge incomparable rows.
    return DatasetSpec(
        name="synth",
        vectors=vectors,
        labels=labels,
        queries=query_vectors,
        ground_truth=gt,
        sha16=_sha16_arrays(vectors, labels, query_vectors, metric.encode("ascii")),
    )


def _load_fvecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise RuntimeError(f"empty fvecs file: {path}")
    dim = int(data[0])
    row_width = dim + 1
    if data.size % row_width != 0:
        raise RuntimeError(f"malformed fvecs file: {path}")
    rows = data.reshape(-1, row_width)
    dims = rows[:, 0]
    if not np.all(dims == dim):
        raise RuntimeError(f"inconsistent fvecs dimensions: {path}")
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def _load_ivecs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise RuntimeError(f"empty ivecs file: {path}")
    dim = int(data[0])
    row_width = dim + 1
    if data.size % row_width != 0:
        raise RuntimeError(f"malformed ivecs file: {path}")
    rows = data.reshape(-1, row_width)
    if not np.all(rows[:, 0] == dim):
        raise RuntimeError(f"inconsistent ivecs dimensions: {path}")
    return np.ascontiguousarray(rows[:, 1:].astype(np.uint64, copy=False))


def _load_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"truncated fbin header: {path}")
        n, dim = struct.unpack("<II", header)
        body = np.fromfile(f, dtype=np.float32, count=n * dim)
    if body.size != n * dim:
        raise RuntimeError(f"truncated fbin body: {path}")
    return np.ascontiguousarray(body.reshape(n, dim))


def _load_ibin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"truncated ibin header: {path}")
        n, dim = struct.unpack("<II", header)
        body = np.fromfile(f, dtype=np.uint32, count=n * dim)
    if body.size != n * dim:
        raise RuntimeError(f"truncated ibin body: {path}")
    return np.ascontiguousarray(body.reshape(n, dim).astype(np.uint64, copy=False))


def load_sift1m(root: Path) -> DatasetSpec:
    root = Path(root)
    base = root / "sift_base.fvecs"
    query = root / "sift_query.fvecs"
    gt = root / "sift_groundtruth.ivecs"
    missing = [p for p in (base, query) if not p.is_file()]
    if missing:
        raise DatasetMissing(f"missing sift1m dataset file(s): {', '.join(str(p) for p in missing)}")

    vectors = _load_fvecs(base)
    queries = _load_fvecs(query)
    ground_truth = _load_ivecs(gt) if gt.is_file() else None
    sha_paths = [base, query] + ([gt] if gt.is_file() else [])
    return DatasetSpec("sift1m", vectors, _labels(vectors.shape[0]), queries, ground_truth, _sha16_files(sha_paths))


def load_gist1m(root: Path) -> DatasetSpec:
    root = Path(root)
    base = root / "gist_base.fbin"
    query = root / "gist_query.fbin"
    gt = root / "gist_gt.ibin"
    missing = [p for p in (base, query) if not p.is_file()]
    if missing:
        raise DatasetMissing(f"missing gist1m dataset file(s): {', '.join(str(p) for p in missing)}")

    vectors = _load_fbin(base)
    queries = _load_fbin(query)
    ground_truth = _load_ibin(gt) if gt.is_file() else None
    sha_paths = [base, query] + ([gt] if gt.is_file() else [])
    return DatasetSpec("gist1m", vectors, _labels(vectors.shape[0]), queries, ground_truth, _sha16_files(sha_paths))


def load_laser_files(
    *,
    n: int,
    dim: int,
    query_count: int,
    seed: int,
    vectors_path: Optional[Path] = None,
    queries_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    laser_src_dir: Optional[Path] = None,
) -> DatasetSpec:
    if vectors_path is not None:
        vectors = _load_fbin(Path(vectors_path))
        n = int(vectors.shape[0])
        dim = int(vectors.shape[1])
    else:
        vectors = np.empty((0, dim), dtype=np.float32)

    labels = (1_000_000 + np.arange(n, dtype=np.uint64)).astype(np.uint64)
    if queries_path is not None:
        queries = _load_fbin(Path(queries_path))
        if queries.shape[1] != dim:
            raise ValueError(f"--queries-path dim ({queries.shape[1]}) does not match LASER dim ({dim})")
    else:
        rng = np.random.default_rng(seed)
        queries = np.ascontiguousarray(rng.standard_normal((query_count, dim)).astype(np.float32))

    if ground_truth_path is not None:
        ground_truth = _load_ibin(Path(ground_truth_path))
    elif vectors_path is not None:
        ground_truth = exact_ground_truth(vectors, labels, queries, metric="L2", k=min(100, n))
    else:
        ground_truth = None

    sha_inputs: list = [labels, queries]
    if vectors_path is not None:
        sha_inputs.insert(0, vectors)
    if ground_truth is not None:
        sha_inputs.append(ground_truth)
    # Fold the prebuilt LASER index directory into the hash. Two runs with
    # identical vectors/queries/GT but distinct prebuilt LASER artifacts
    # (e.g. different graph degree, different sharding) MUST report
    # different `dataset_sha256_prefix` values — they are not the same
    # input from a recall-reproducibility standpoint.
    if laser_src_dir is not None and Path(laser_src_dir).is_dir():
        sha_inputs.append(b"laser_src_dir=" + _sha16_directory(Path(laser_src_dir)).encode("ascii"))

    return DatasetSpec(
        "laser_files",
        vectors,
        labels,
        np.ascontiguousarray(queries, dtype=np.float32),
        ground_truth,
        _sha16_arrays(*sha_inputs),
        row_count=n,
        dimension=dim,
    )
