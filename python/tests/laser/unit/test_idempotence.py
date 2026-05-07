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

"""Tests for LASER index build idempotence helpers."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from alayalite.laser._idempotence import (
    validate_laser_index,
    validate_medoids,
    validate_pca_base,
    validate_pca_params,
    validate_vamana_index,
)


def _write_fbin(path: Path, n: int, dim: int, *, truncate_data_bytes: int = 0) -> None:
    with path.open("wb") as f:
        np.array([n, dim], dtype=np.int32).tofile(f)
        payload = np.arange(n * dim, dtype=np.float32).tobytes()
        if truncate_data_bytes > 0:
            payload = payload[:-truncate_data_bytes]
        f.write(payload)


def _write_vamana_index(path: Path, *, R: int, expected_size: int | None = None, frozen_pts: int = 0) -> None:
    real_size = expected_size if expected_size is not None else 24 + 16
    header = (
        struct.pack("<Q", real_size)
        + struct.pack("<I", int(R))
        + struct.pack("<I", 0)
        + struct.pack("<Q", int(frozen_pts))
    )
    with path.open("wb") as f:
        f.write(header)
        if real_size > 24:
            f.write(b"\x00" * (real_size - 24))


def _write_pca_params(path: Path, dim: int, *, truncate_bytes: int = 0) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", dim))
        payload = np.arange(dim + dim * dim, dtype=np.float32).tobytes()
        if truncate_bytes > 0:
            payload = payload[:-truncate_bytes]
        f.write(payload)


def test_validate_vamana_index_happy_and_mismatch_and_truncated_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "graph.index"
    assert not validate_vamana_index(str(path), 64)

    path.write_bytes(b"\x00" * 8)
    assert not validate_vamana_index(str(path), 64)

    _write_vamana_index(path, R=64)
    assert validate_vamana_index(str(path), 64)
    assert not validate_vamana_index(str(path), 32)

    _write_vamana_index(path, R=64, frozen_pts=1)
    assert not validate_vamana_index(str(path), 64)


def test_validate_pca_base_happy_and_mismatch_and_truncated_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "x_pca_base.fbin"
    assert not validate_pca_base(str(path), 4, 3)

    _write_fbin(path, 4, 3, truncate_data_bytes=4)
    assert not validate_pca_base(str(path), 4, 3)

    _write_fbin(path, 4, 3)
    assert validate_pca_base(str(path), 4, 3)
    assert not validate_pca_base(str(path), 4, 4)


def test_validate_pca_params_happy_and_mismatch_and_truncated_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "x_pca.bin"
    assert not validate_pca_params(str(path), 8)

    _write_pca_params(path, 8, truncate_bytes=4)
    assert not validate_pca_params(str(path), 8)

    _write_pca_params(path, 8)
    assert validate_pca_params(str(path), 8)
    assert not validate_pca_params(str(path), 16)


def test_validate_medoids_happy_and_missing_and_empty(tmp_path: Path) -> None:
    prefix = tmp_path / "x"
    assert not validate_medoids(str(prefix))

    (tmp_path / "x_medoids").write_bytes(b"\x00")
    assert not validate_medoids(str(prefix))

    (tmp_path / "x_medoids_indices").write_bytes(b"")
    assert not validate_medoids(str(prefix))

    (tmp_path / "x_medoids_indices").write_bytes(b"\x01")
    assert validate_medoids(str(prefix))


def test_validate_laser_index_happy_and_mismatch_and_truncated_and_missing(tmp_path: Path) -> None:
    prefix = tmp_path / "x"
    path = tmp_path / "x_R64_MD128.index"
    assert not validate_laser_index(str(prefix), 64, 128, 123)

    path.write_bytes(struct.pack("<I", 123))
    assert not validate_laser_index(str(prefix), 64, 128, 123)

    path.write_bytes(struct.pack("<Q", 123) + b"\x00" * 16)
    assert validate_laser_index(str(prefix), 64, 128, 123)
    assert not validate_laser_index(str(prefix), 64, 128, 124)
