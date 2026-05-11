# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Artifact validators used by ``alayalite.laser.Index.fit(skip_existing=True)``."""

from __future__ import annotations

import os
import struct

_VAMANA_HEADER_BYTES = 24
_SEED_SIDECAR_SUFFIX = "_seed.txt"


def _exists_non_empty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def validate_vamana_index(path: str, graph_r: int) -> bool:
    """Validate a DiskANN single-file graph header at ``path``."""
    try:
        file_size = os.path.getsize(path)
        if file_size < _VAMANA_HEADER_BYTES:
            return False
        with open(path, "rb") as f:
            head = f.read(_VAMANA_HEADER_BYTES)
        if len(head) != _VAMANA_HEADER_BYTES:
            return False
        expected_size = struct.unpack("<Q", head[0:8])[0]
        max_degree = struct.unpack("<I", head[8:12])[0]
        frozen_pts = struct.unpack("<Q", head[16:24])[0]
        return file_size == expected_size and max_degree == int(graph_r) and frozen_pts == 0
    except (OSError, struct.error):
        return False


def validate_pca_base(path: str, n: int, dim: int) -> bool:
    """Validate ``<prefix>_pca_base.fbin`` header and payload size."""
    try:
        size = os.path.getsize(path)
        expected = 8 + int(n) * int(dim) * 4
        if size != expected:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) != 8:
            return False
        got_n, got_dim = struct.unpack("<ii", head)
        return got_n == int(n) and got_dim == int(dim)
    except (OSError, struct.error):
        return False


def validate_pca_params(path: str, dim: int) -> bool:
    """Validate ``save_pca_params`` binary layout."""
    try:
        size = os.path.getsize(path)
        d = int(dim)
        expected = 8 + d * 4 + d * d * 4
        if size != expected:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) != 8:
            return False
        (declared_dim,) = struct.unpack("<Q", head)
        return declared_dim == d
    except (OSError, struct.error):
        return False


def validate_medoids(prefix: str) -> bool:
    """Validate ``<prefix>_medoids`` and ``<prefix>_medoids_indices`` presence."""
    return _exists_non_empty(f"{prefix}_medoids") and _exists_non_empty(f"{prefix}_medoids_indices")


def validate_laser_index(prefix: str, graph_r: int, main_dim: int, n: int) -> bool:
    """Validate LASER main index presence and leading-qword count."""
    path = f"{prefix}_R{int(graph_r)}_MD{int(main_dim)}.index"
    try:
        if not _exists_non_empty(path):
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) != 8:
            return False
        (declared_n,) = struct.unpack("<Q", head)
        return declared_n == int(n)
    except (OSError, struct.error):
        return False


def seed_sidecar_path(prefix: str) -> str:
    """Return the seed-sidecar path used by the validator and writer."""
    return f"{prefix}{_SEED_SIDECAR_SUFFIX}"


def validate_seed_sidecar(prefix: str, seed: int) -> bool:
    """Return True iff ``<prefix>_seed.txt`` exists and matches ``seed``.

    The per-artifact validators above only check size + header, so they
    cannot detect that an existing artifact was built from a different
    master seed. Without this gate, ``Index.fit(seed=Y, skip_existing=True)``
    would silently reuse artifacts built earlier with ``seed=X`` and the
    caller would think they got a seed-Y index. A missing sidecar is
    treated as mismatch so legacy artifacts produced before this contract
    do not silently leak into a new seed's run.
    """
    path = seed_sidecar_path(prefix)
    try:
        if not _exists_non_empty(path):
            return False
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        return int(content) == int(seed)
    except (OSError, ValueError):
        return False


def invalidate_seed_sidecar(prefix: str) -> None:
    """Remove ``<prefix>_seed.txt`` if present before rebuilding mismatched artifacts."""
    try:
        os.remove(seed_sidecar_path(prefix))
    except FileNotFoundError:
        pass


def write_seed_sidecar(prefix: str, seed: int) -> None:
    """Atomically publish ``<prefix>_seed.txt`` with the master seed.

    Written via tmp-file + ``os.replace`` so a crash mid-write leaves the
    previous sidecar intact rather than producing a half-written file
    that ``validate_seed_sidecar`` would reject as a parse error.
    """
    target = seed_sidecar_path(prefix)
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{int(seed)}\n")
    os.replace(tmp, target)
