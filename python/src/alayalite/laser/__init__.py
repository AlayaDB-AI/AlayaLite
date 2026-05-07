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

from alayalite._alayalitepy import laser as _raw_laser_mod  # type: ignore[attr-defined]

from ._idempotence import (
    validate_laser_index,
    validate_medoids,
    validate_pca_base,
    validate_pca_params,
    validate_vamana_index,
)
from ._io import write_fbin
from ._seeds import derive_subseeds

PathLikeStr = Union[str, os.PathLike[str]]
_INDEX_RE = re.compile(r"_R(?P<R>\d+)_MD(?P<md>\d+)\.index$")


@dataclasses.dataclass(frozen=True)
class _IndexParams:
    metric: str
    n: int
    raw_dim: int
    main_dim: int
    R: int  # pylint: disable=invalid-name
    prefix: str


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
    if main_dim <= 0 or (main_dim & (main_dim - 1)) != 0:
        raise ValueError(f"main_dim must be a positive power of two, got {main_dim}")
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
        *,
        output_dir: PathLikeStr,
        name: str = "laser",
        metric: str = "l2",
        main_dim: int | None = None,
        R: int = 64,
        L: int = 100,
        alpha: float = 1.2,
        ef_indexing: int = 200,
        beam_width: int = 16,
        num_threads: int = 0,
        ep_num: int = 300,
        seed: int = 42,
        pca_seed: int | None = None,
        medoid_seed: int | None = None,
        vamana_seed: int | None = None,
        rotator_seed: int | None = None,
        dram_budget_gb: float = 1.0,
        disable_medoid: bool = False,
        skip_existing: bool = True,
        auto_load: bool = True,
    ) -> Index:
        """Build a LASER index in one call and optionally return it search-ready.

        Notes
        -----
        When ``main_dim == raw_dim`` (or ``main_dim is None``), PCA is skipped and
        ``<prefix>_pca.bin`` is intentionally absent. The C++ load path may print
        ``Warning: PCA file not found: ...`` to stderr; this is expected behaviour.
        """
        metric = _canonical_metric(metric)
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
        resolved_main_dim = raw_dim if main_dim is None else int(main_dim)
        _validate_main_dim(resolved_main_dim, raw_dim)

        sub = derive_subseeds(int(seed))
        pca_seed = int(sub.pca if pca_seed is None else pca_seed)
        medoid_seed = int(sub.medoid if medoid_seed is None else medoid_seed)
        vamana_seed = int(sub.vamana if vamana_seed is None else vamana_seed)
        rotator_seed = int(sub.rotator if rotator_seed is None else rotator_seed)

        if vectors_arr is not None:
            if not (skip_existing and validate_pca_base(raw_fbin_path, n, raw_dim)):
                write_fbin(raw_fbin_path, vectors_arr)

        pca_base_path = f"{prefix}_pca_base.fbin"
        pca_params_path = f"{prefix}_pca.bin"
        if resolved_main_dim < raw_dim:
            if not (
                skip_existing
                and validate_pca_base(pca_base_path, n, raw_dim)
                and validate_pca_params(pca_params_path, raw_dim)
            ):
                from alayalite.laser._pca import (  # pylint: disable=import-outside-toplevel
                    fit_incremental_pca,
                    pca_transform_and_save,
                    sample_vectors_from_fbin,
                    save_pca_params,
                )

                vectors, sample_vectors = sample_vectors_from_fbin(raw_fbin_path, seed=pca_seed)
                pca = fit_incremental_pca(sample_vectors, n_components=raw_dim)
                save_pca_params(pca, pca_params_path)
                pca_transform_and_save(vectors, pca, pca_base_path)
        else:
            if not (skip_existing and validate_pca_base(pca_base_path, n, raw_dim)):
                shutil.copyfile(raw_fbin_path, pca_base_path)
            # Keep the no-PCA branch explicit: stale files would rotate queries unexpectedly.
            if os.path.exists(pca_params_path):
                os.remove(pca_params_path)

        if not disable_medoid:
            if not (skip_existing and validate_medoids(prefix)):
                from alayalite.laser._medoid import generate_and_save_medoids  # pylint: disable=import-outside-toplevel

                generate_and_save_medoids(
                    pca_base_path,
                    f"{prefix}_medoids_indices",
                    f"{prefix}_medoids",
                    int(ep_num),
                    seed=medoid_seed,
                )

        vamana_path = f"{prefix}_vamana_graph.index"
        if not (skip_existing and validate_vamana_index(vamana_path, int(R))):
            from alayalite import vamana as vamana_mod  # pylint: disable=import-outside-toplevel

            vamana_mod.build_index(
                data_path=raw_fbin_path,
                output_path=vamana_path,
                R=int(R),
                L=int(L),
                alpha=float(alpha),
                seed=vamana_seed,
                num_threads=int(num_threads),
                dram_budget_gb=float(dram_budget_gb),
            )

        raw = _raw_laser_mod.Index(
            index_type="QG",
            metric=metric,
            num_elements=int(n),
            main_dimension=int(resolved_main_dim),
            dimension=int(raw_dim),
            degree_bound=int(R),
            rotator_seed=rotator_seed,
            rotator_dump_path="",
        )
        if not (skip_existing and validate_laser_index(prefix, int(R), int(resolved_main_dim), int(n))):
            raw.build_index(
                vamana_file=vamana_path,
                data_file=prefix,
                EF=int(ef_indexing),
                num_thread=int(num_threads),
            )

        loaded = False
        if auto_load:
            raw.load(prefix, float(dram_budget_gb))
            raw.set_params(
                ef_search=int(ef_indexing),
                num_threads=_effective_threads(int(num_threads)),
                beam_width=int(beam_width),
            )
            loaded = True

        params = _IndexParams(
            metric=metric,
            n=int(n),
            raw_dim=int(raw_dim),
            main_dim=int(resolved_main_dim),
            R=int(R),
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

        raw = _raw_laser_mod.Index(
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
        raw.set_params(ef_search=200, num_threads=_effective_threads(0), beam_width=16)

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
        return self._raw.search(q, int(k))

    def batch_search(self, queries: np.ndarray, k: int):
        self._require_loaded()
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError(f"expected 2D queries, got ndim={q.ndim}")
        return self._raw.batch_search(q, int(k))

    def set_params(self, ef_search: int = 200, num_threads: int = 0, beam_width: int = 16) -> None:
        self._require_loaded()
        self._raw.set_params(
            ef_search=int(ef_search),
            num_threads=_effective_threads(int(num_threads)),
            beam_width=int(beam_width),
        )


__all__ = ["Index"]
