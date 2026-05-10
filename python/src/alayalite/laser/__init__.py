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

"""Laser on-disk Quantized Graph index public API.

The raw pybind class still lives at ``alayalite._alayalitepy.laser.Index``.
This module exposes a higher-level wrapper with a unified build entrypoint:
``alayalite.laser.Index.fit(...)``.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Union

import numpy as np

try:
    from alayalite._alayalitepy import laser as _raw_laser_mod  # type: ignore[attr-defined]
except (AttributeError, ImportError):
    _raw_laser_mod = None

from ._idempotence import (
    invalidate_seed_sidecar,
    validate_laser_index,
    validate_medoids,
    validate_pca_base,
    validate_pca_params,
    validate_seed_sidecar,
    validate_vamana_index,
    write_seed_sidecar,
)
from ._io import write_fbin

PathLikeStr = Union[str, os.PathLike[str]]
_INDEX_RE = re.compile(r"_R(?P<R>\d+)_MD(?P<md>\d+)\.index$")


@dataclasses.dataclass(frozen=True)
class BuildParams:
    """LASER build-time hyperparameters.

    Grouped to keep ``Index.fit`` signature flat at the call site. Adding a new
    build knob means adding a field here, not extending ``fit``.

    Fields
    ------
    metric : "l2"
        Distance metric used by the underlying graph and quantizer.
    main_dim : int | None
        PCA target dimension. ``None`` means "no PCA"; the raw dimension is used.
        Must be a power of two ``>= 128`` and ``<= raw_dim`` when set.
    R : int
        Vamana out-degree bound (also the LASER graph degree).
    L : int
        Vamana search-list size during graph construction.
    alpha : float
        Vamana RNG-prune slack factor.
    ef_indexing : int
        EF used during the LASER quantization pass over the Vamana graph.
    ep_num : int
        Number of medoid entry points generated for search seeding.
    disable_medoid : bool
        Skip medoid generation entirely (search uses the graph's start node).
    """

    metric: str = "l2"
    main_dim: int | None = None
    R: int = 64  # pylint: disable=invalid-name
    L: int = 200  # pylint: disable=invalid-name
    alpha: float = 1.2
    ef_indexing: int = 200
    ep_num: int = 300
    disable_medoid: bool = False


@dataclasses.dataclass(frozen=True)
class _IndexParams:
    metric: str
    n: int
    raw_dim: int
    main_dim: int
    R: int  # pylint: disable=invalid-name
    prefix: str


def _require_raw_laser():
    if _raw_laser_mod is None:
        raise RuntimeError("alayalite.laser native bindings are not available in this build")
    return _raw_laser_mod


def _canonical_metric(metric: str) -> str:
    normalized = str(metric).lower()
    if normalized in {"l2", "euclidean"}:
        return "l2"
    raise ValueError(f"LASER supports metric='l2' only, got {metric!r}")


def _effective_threads(num_threads: int) -> int:
    if int(num_threads) > 0:
        return int(num_threads)
    return os.cpu_count() or 1


def _read_fbin_header(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) != 8:
        raise ValueError(f"invalid fbin header (need 8 bytes): {path}")
    n, dim = struct.unpack("<ii", head)
    if n <= 0 or dim <= 0:
        raise ValueError(f"invalid fbin shape ({n}, {dim}) in {path}")
    size = os.path.getsize(path)
    expected = 8 + n * dim * 4
    if size != expected:
        raise ValueError(f"invalid fbin payload size for {path}: expected {expected}, got {size}")
    return int(n), int(dim)


def _parse_index_filename(path: str) -> tuple[int, int]:
    name = Path(path).name
    m = _INDEX_RE.search(name)
    if m is None:
        raise ValueError(f"cannot parse R/MD from LASER index filename: {name}")
    return int(m.group("R")), int(m.group("md"))


def _read_index_header_24(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) != 24:
        raise ValueError(f"LASER index file too short for 24-byte header: {path}")
    n, dim, _ = struct.unpack("<QQQ", head)
    return int(n), int(dim)


def _validate_main_dim(main_dim: int, raw_dim: int) -> None:
    # Order matters: 0 & -1 == 0 in Python's bigint, so the power-of-two
    # check would falsely pass for main_dim == 0 without this guard.
    if main_dim <= 0:
        raise ValueError(f"main_dim must be positive, got {main_dim}")
    if main_dim & (main_dim - 1) != 0:
        raise ValueError(f"main_dim must be a power of two, got {main_dim}")
    if main_dim < 128:
        raise ValueError(f"main_dim must be >= 128 (LASER floor), got {main_dim}")
    if main_dim > raw_dim:
        raise ValueError(f"main_dim ({main_dim}) must be <= raw_dim ({raw_dim})")


class Index:
    """Unified Python wrapper around ``alayalite._alayalitepy.laser.Index``."""

    def __init__(self, raw, prefix: str, params: _IndexParams, *, loaded: bool) -> None:
        self._raw = raw
        self._prefix = str(prefix)
        self._params = params
        self._loaded = bool(loaded)

    @classmethod
    def fit(
        cls,
        vectors_or_fbin,
        output_dir: PathLikeStr,
        *,
        name: str = "laser",
        build_params: BuildParams | None = None,
        seed: int = 42,
        num_threads: int = 0,
        dram_budget_gb: float = 1.0,
        skip_existing: bool = True,
        auto_load: bool = True,
    ) -> Index:
        """Build a LASER index in one call and optionally return it search-ready.

        Parameters
        ----------
        vectors_or_fbin
            Either an in-memory ``float32`` ndarray of shape ``(n, raw_dim)`` or
            a path to a DiskANN-style ``.fbin`` file.
        output_dir
            Directory used for all build artifacts (PCA, medoids, vamana, index).
        name
            Filename prefix inside ``output_dir`` (e.g. ``"laser"`` ->
            ``laser_pca.bin``, ``laser_R64_MD256.index``).
        build_params
            All build-time hyperparameters. ``None`` uses ``BuildParams()`` defaults.
        seed
            Master RNG seed shared by every randomized build sub-step
            (PCA sampling, medoid clustering, Vamana, LASER rotator).
            Search-time settings live on the returned ``Index``; use
            ``set_params(...)`` after ``fit``.
        skip_existing
            When True, reuse on-disk artifacts whose size + header match the
            requested shape AND whose seed sidecar (``<prefix>_seed.txt``)
            matches ``seed``. A missing sidecar (e.g. legacy artifacts built
            before this contract) forces a rebuild — the per-artifact
            validators only check size/header, so without the sidecar gate
            ``fit(seed=Y, skip_existing=True)`` would silently reuse
            artifacts built earlier with ``seed=X``.

        Notes
        -----
        When ``build_params.main_dim is None`` or equals the raw dimension, PCA is
        skipped and ``<prefix>_pca.bin`` is intentionally absent. The C++ load
        path may print ``Warning: PCA file not found: ...`` to stderr; this is
        expected behaviour.
        """
        bp = build_params if build_params is not None else BuildParams()
        metric = _canonical_metric(bp.metric)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prefix = os.fspath(output / str(name))

        vectors_arr: np.ndarray | None = None
        if isinstance(vectors_or_fbin, (str, os.PathLike)):
            raw_fbin_path = os.fspath(vectors_or_fbin)
            if not os.path.isfile(raw_fbin_path):
                raise FileNotFoundError(f"raw fbin file not found: {raw_fbin_path}")
            n, raw_dim = _read_fbin_header(raw_fbin_path)
        else:
            vectors_arr = np.asarray(vectors_or_fbin)
            if vectors_arr.dtype != np.float32:
                raise TypeError(f"expected dtype float32, got {vectors_arr.dtype}")
            if vectors_arr.ndim != 2:
                raise ValueError(f"expected 2D array, got ndim={vectors_arr.ndim}")
            vectors_arr = np.ascontiguousarray(vectors_arr, dtype=np.float32)
            n, raw_dim = int(vectors_arr.shape[0]), int(vectors_arr.shape[1])
            raw_fbin_path = f"{prefix}_raw.fbin"

        if raw_dim < 128:
            raise ValueError(f"LASER requires raw_dim >= 128, got {raw_dim}")
        resolved_main_dim = raw_dim if bp.main_dim is None else int(bp.main_dim)
        _validate_main_dim(resolved_main_dim, raw_dim)

        master_seed = int(seed)
        resolved_threads = _effective_threads(int(num_threads))

        # Sidecar gate: a missing-or-mismatched seed sidecar disables
        # skip_existing for this call so all sub-steps rebuild against the
        # requested master_seed. The sidecar is republished at the end of
        # fit() once every build step has succeeded, so a crash mid-build
        # leaves the previous (or absent) sidecar in place — the next
        # invocation will then see "mismatch" and rebuild any partially
        # written artifacts rather than trusting them.
        seed_matches = validate_seed_sidecar(prefix, master_seed)
        if not seed_matches:
            invalidate_seed_sidecar(prefix)
        effective_skip = bool(skip_existing) and seed_matches

        if vectors_arr is not None:
            if not (effective_skip and validate_pca_base(raw_fbin_path, n, raw_dim)):
                write_fbin(raw_fbin_path, vectors_arr)

        pca_base_path = f"{prefix}_pca_base.fbin"
        pca_params_path = f"{prefix}_pca.bin"
        if resolved_main_dim < raw_dim:
            if not (
                effective_skip
                and validate_pca_base(pca_base_path, n, raw_dim)
                and validate_pca_params(pca_params_path, raw_dim)
            ):
                from alayalite.laser._pca import (  # pylint: disable=import-outside-toplevel
                    fit_incremental_pca,
                    pca_transform_and_save,
                    sample_vectors_from_fbin,
                    save_pca_params,
                )

                vectors, sample_vectors = sample_vectors_from_fbin(raw_fbin_path, seed=master_seed)
                pca = fit_incremental_pca(sample_vectors, n_components=raw_dim)
                save_pca_params(pca, pca_params_path)
                pca_transform_and_save(vectors, pca, pca_base_path)
        else:
            if not (effective_skip and validate_pca_base(pca_base_path, n, raw_dim)):
                shutil.copyfile(raw_fbin_path, pca_base_path)
            # Keep the no-PCA branch explicit: stale files would rotate queries unexpectedly.
            if os.path.exists(pca_params_path):
                os.remove(pca_params_path)

        if not bp.disable_medoid:
            if not (effective_skip and validate_medoids(prefix)):
                from alayalite.laser._medoid import generate_and_save_medoids  # pylint: disable=import-outside-toplevel

                generate_and_save_medoids(
                    pca_base_path,
                    f"{prefix}_medoids_indices",
                    f"{prefix}_medoids",
                    int(bp.ep_num),
                    seed=master_seed,
                )

        vamana_path = f"{prefix}_vamana_graph.index"
        if not (effective_skip and validate_vamana_index(vamana_path, int(bp.R))):
            from alayalite import vamana as vamana_mod  # pylint: disable=import-outside-toplevel

            vamana_mod.build_index(
                data_path=raw_fbin_path,
                output_path=vamana_path,
                R=int(bp.R),
                L=int(bp.L),
                alpha=float(bp.alpha),
                seed=master_seed,
                num_threads=resolved_threads,
                dram_budget_gb=float(dram_budget_gb),
            )

        raw_laser_mod = _require_raw_laser()
        raw = raw_laser_mod.Index(
            index_type="QG",
            metric=metric,
            num_elements=int(n),
            main_dimension=int(resolved_main_dim),
            dimension=int(raw_dim),
            degree_bound=int(bp.R),
            rotator_seed=master_seed,
            rotator_dump_path="",
        )
        if not (effective_skip and validate_laser_index(prefix, int(bp.R), int(resolved_main_dim), int(n))):
            raw.build_index(
                vamana_file=vamana_path,
                data_file=prefix,
                EF=int(bp.ef_indexing),
                num_thread=resolved_threads,
            )

        write_seed_sidecar(prefix, master_seed)

        loaded = False
        if auto_load:
            raw.load(prefix, float(dram_budget_gb))
            # Sane defaults so a freshly-built index is searchable without an
            # explicit set_params() call. Override via Index.set_params(...).
            raw.set_params(
                ef_search=int(bp.ef_indexing),
                num_threads=resolved_threads,
                beam_width=16,
            )
            loaded = True

        params = _IndexParams(
            metric=metric,
            n=int(n),
            raw_dim=int(raw_dim),
            main_dim=int(resolved_main_dim),
            R=int(bp.R),
            prefix=prefix,
        )
        return cls(raw=raw, prefix=prefix, params=params, loaded=loaded)

    @classmethod
    def from_prefix(cls, prefix: PathLikeStr, dram_budget_gb: float = 1.0) -> Index:
        """Load an existing LASER index from a prefix (without ``_R*_MD*.index``)."""
        base = os.fspath(prefix)
        pattern = f"{base}_R*_MD*.index"
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no LASER index file found; searched pattern: {pattern}")
        if len(matches) > 1:
            raise ValueError(f"multiple LASER index files match prefix {base!r}: {[Path(m).name for m in matches]}")

        index_path = matches[0]
        file_graph_r, file_main_dim = _parse_index_filename(index_path)
        n, header_main_dim = _read_index_header_24(index_path)
        if header_main_dim and header_main_dim != file_main_dim:
            raise ValueError(
                f"index header/file-name MD mismatch for {index_path}: header={header_main_dim}, file={file_main_dim}"
            )

        raw_dim = file_main_dim
        pca_base = f"{base}_pca_base.fbin"
        if os.path.exists(pca_base):
            base_n, base_dim = _read_fbin_header(pca_base)
            if base_n == n and base_dim >= file_main_dim:
                raw_dim = base_dim

        raw_laser_mod = _require_raw_laser()
        raw = raw_laser_mod.Index(
            index_type="QG",
            metric="l2",
            num_elements=int(n),
            main_dimension=int(file_main_dim),
            dimension=int(raw_dim),
            degree_bound=int(file_graph_r),
            rotator_seed=0,
            rotator_dump_path="",
        )
        raw.load(base, float(dram_budget_gb))

        params = _IndexParams(
            metric="l2",
            n=int(n),
            raw_dim=int(raw_dim),
            main_dim=int(file_main_dim),
            R=int(file_graph_r),
            prefix=base,
        )
        return cls(raw=raw, prefix=base, params=params, loaded=True)

    @property
    def prefix(self) -> str:
        return self._prefix

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "LASER index is not loaded. Use Index.fit(..., auto_load=True) or Index.from_prefix(...)."
            )

    def search(self, query: np.ndarray, k: int):
        self._require_loaded()
        q = np.ascontiguousarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"expected 1D query, got ndim={q.ndim}")
        if q.shape[0] != self._params.raw_dim:
            raise ValueError(f"expected query.shape[0]={self._params.raw_dim}, got query.shape[0]={q.shape[0]}")
        return self._raw.search(q, int(k))

    def batch_search(self, queries: np.ndarray, k: int):
        self._require_loaded()
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError(f"expected 2D queries, got ndim={q.ndim}")
        if q.shape[1] != self._params.raw_dim:
            raise ValueError(f"expected queries.shape[1]={self._params.raw_dim}, got queries.shape[1]={q.shape[1]}")
        return self._raw.batch_search(q, int(k))

    def set_params(self, ef_search: int = 200, num_threads: int = 0, beam_width: int = 16) -> None:
        self._require_loaded()
        self._raw.set_params(
            ef_search=int(ef_search),
            num_threads=_effective_threads(int(num_threads)),
            beam_width=int(beam_width),
        )


# Re-exported raw pybind class for callers that need step-by-step construction
# (e.g. the disk-test LASER fixture). Use Index.fit() above for normal usage.
RawIndex = _raw_laser_mod.Index if _raw_laser_mod is not None else None


__all__ = ["BuildParams", "Index", "RawIndex"]
