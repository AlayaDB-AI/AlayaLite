"""TOML-driven configuration for the LASER cross-benchmark experiments.

Every step (build, bench, plot) accepts ``--config <path>``; all paths,
hyperparameters and the Vamana-source matrix come from a single TOML file
under ``configs/``. No hardcoded paths in the step scripts.

TOML rather than YAML so the config is readable by both AlayaLite and the
upstream Laser venv via stdlib ``tomllib`` (no pyyaml dependency).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import tomllib

LaserKind = Literal["orig", "lite"]
BuilderKind = Literal["diskann", "alayalite"]


@dataclasses.dataclass(frozen=True)
class Paths:
    data_dir: Path
    tmp_dir: Path
    # Optional: only required by the steps that actually invoke them.
    # Missing values surface as None and the consumer raises a precise error.
    laser_venv: Path | None = None
    diskann_binary: Path | None = None

    def require_laser_venv(self) -> Path:
        if self.laser_venv is None:
            raise ValueError("config.paths.laser_venv is unset; required to run step3 (orig LASER pipeline)")
        return self.laser_venv

    def require_diskann_binary(self) -> Path:
        if self.diskann_binary is None:
            raise ValueError("config.paths.diskann_binary is unset; required to run step1 (DiskANN Vamana build)")
        return self.diskann_binary


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Resolves dataset filenames from a single ``prefix``.

    Convention: ``<prefix>_base.fbin``, ``<prefix>_query.fbin``, ``<prefix>_gt.ibin``.
    Override via the explicit fields when the dataset breaks the convention.
    """

    prefix: str
    base_filename: str | None = None
    query_filename: str | None = None
    gt_filename: str | None = None

    def base_fbin(self, data_dir: Path) -> Path:
        return data_dir / (self.base_filename or f"{self.prefix}_base.fbin")

    def query_fbin(self, data_dir: Path) -> Path:
        return data_dir / (self.query_filename or f"{self.prefix}_query.fbin")

    def gt_ibin(self, data_dir: Path) -> Path:
        return data_dir / (self.gt_filename or f"{self.prefix}_gt.ibin")


@dataclasses.dataclass(frozen=True)
class BuildKnobs:
    """LASER build-time hyperparameters (mirrors ``alayalite.laser.BuildParams``)."""

    R: int = 64  # pylint: disable=invalid-name
    main_dim: int = 256
    L: int = 100  # pylint: disable=invalid-name
    alpha: float = 1.2
    ef_indexing: int = 200
    ep_num: int = 300


@dataclasses.dataclass(frozen=True)
class VamanaSource:
    """One row of the ``vamana_sources`` matrix.

    ``tag`` names the produced Vamana graph (used in artifact paths and CSV names).
    ``builder`` selects step1 (``diskann``) or step2 (``alayalite``).
    ``L`` is the search-list size; for DiskANN this is the build-time L.
    """

    tag: str
    builder: BuilderKind
    L: int  # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class BenchKnobs:
    """Search-time bench parameters (apply to both orig and lite cells)."""

    k: int = 10
    beam: int = 16
    warmup: int = 10
    runs: int = 30
    ef_sweep: tuple[int, ...] = (80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500)
    build_threads: int = 48
    search_threads: int = 1
    numactl: bool = True
    seed: int = 42
    dram_budget_gb: float = 1.0


@dataclasses.dataclass(frozen=True)
class CrossBenchConfig:
    name: str
    paths: Paths
    dataset: Dataset
    build: BuildKnobs
    vamana_sources: tuple[VamanaSource, ...]
    bench: BenchKnobs

    @classmethod
    def from_toml(cls, path: Path | str) -> CrossBenchConfig:
        path = Path(path).expanduser().resolve()
        with path.open("rb") as f:
            raw = tomllib.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level TOML must be a table, got {type(raw).__name__}")

        paths_raw = raw.get("paths") or {}
        bench_raw = dict(raw.get("bench") or {})
        if "ef_sweep" in bench_raw:
            bench_raw["ef_sweep"] = tuple(int(x) for x in bench_raw["ef_sweep"])

        return cls(
            name=str(raw["name"]),
            paths=Paths(
                data_dir=Path(paths_raw["data_dir"]).expanduser(),
                tmp_dir=Path(paths_raw["tmp_dir"]).expanduser(),
                laser_venv=(Path(paths_raw["laser_venv"]).expanduser() if paths_raw.get("laser_venv") else None),
                diskann_binary=(
                    Path(paths_raw["diskann_binary"]).expanduser() if paths_raw.get("diskann_binary") else None
                ),
            ),
            dataset=Dataset(**(raw.get("dataset") or {})),
            build=BuildKnobs(**(raw.get("build") or {})),
            vamana_sources=tuple(VamanaSource(**vs) for vs in raw["vamana_sources"]),
            bench=BenchKnobs(**bench_raw),
        )

    # ── path resolvers ───────────────────────────────────────────────────────

    def vamana_dir(self, source: VamanaSource) -> Path:
        return self.paths.tmp_dir / f"vamana_{source.tag}"

    def vamana_path(self, source: VamanaSource) -> Path:
        return self.vamana_dir(source) / f"{self.dataset.prefix}_vamana.index"

    def find_vamana(self, tag: str) -> VamanaSource:
        for vs in self.vamana_sources:
            if vs.tag == tag:
                return vs
        known = ", ".join(vs.tag for vs in self.vamana_sources)
        raise ValueError(f"unknown vamana_tag={tag!r}; known tags: [{known}]")

    def cell_dir(self, laser: LaserKind, vamana_tag: str) -> Path:
        return self.paths.tmp_dir / f"{laser}_{vamana_tag}"

    def cell_csv(self, laser: LaserKind, vamana_tag: str) -> Path:
        return self.paths.tmp_dir / "results" / f"{laser}_{vamana_tag}.csv"

    def cell_index_prefix(self, laser: LaserKind, vamana_tag: str) -> str:
        """Filesystem prefix used by the LASER index for one cell.

        Two flavours so the orig (legacy) and lite (Index.fit) artifact layouts
        do not collide when writing or comparing md5s side-by-side.
        """
        cell = self.cell_dir(laser, vamana_tag)
        if laser == "orig":
            return str(cell / f"dsqg_{self.dataset.prefix}")
        return str(cell / "laser")

    def results_dir(self) -> Path:
        return self.paths.tmp_dir / "results"
