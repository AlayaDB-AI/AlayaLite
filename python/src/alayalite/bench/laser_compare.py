# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Paired LASER native vs DiskCollection(disk_laser) benchmark harness.

Run with:

    python -m alayalite.bench.laser_compare \\
        --laser-src-dir <dir> --vectors <fbin> --queries-path <fbin> \\
        --ground-truth <ibin> --out results

Implements OpenSpec change ``python-laser-native-equivalence-benchmark``
(capability ``laser-native-equivalence-benchmark``). Loads one precomputed
LASER artifact directory and runs the same query batch against both the
native ``alayalite.laser.Index`` path and the
``DiskCollection(index_type='disk_laser')`` path under matched
``(top_k, ef_search, beam_width, num_threads, queries, warmup, seed,
dram_budget_gb)`` parameters, then emits a paired ``comparison`` JSON
block that quantifies the wrapper-layer adapter overhead.
"""

# pylint: disable=inconsistent-quotes  # Python 3.10 forbids same-quote
#                                        nesting in f-strings; subscript
#                                        access in f-strings keeps single
#                                        quotes inside.
# pylint: disable=invalid-name  # `R`, `MD`, `N` are deliberate one-letter
#                                  symbols matching the C++ LASER
#                                  vocabulary (`degree_bound`,
#                                  `main_dimension`, `num_elements`); the
#                                  filename pattern itself encodes them
#                                  verbatim. Renaming here would diverge
#                                  from the on-disk artifact contract.
# pylint: disable=too-many-locals  # The two adapter functions and the
#                                    main pipeline carry the full
#                                    `(top_k, ef, beam_width, num_threads,
#                                    queries, warmup, seed, dram_budget_gb)`
#                                    parameter set plus build/measure
#                                    bookkeeping; splitting further would
#                                    fragment the timing critical path.
# pylint: disable=too-many-arguments,too-many-positional-arguments
#                                    # `_write_to_run_dir` and
#                                    `_build_summary` keep their flat
#                                    signatures because each named arg
#                                    maps 1-to-1 to a JSON field; using
#                                    a config object would obscure the
#                                    schema-vs-arg mapping.
# pylint: disable=too-many-lines  # The module bundles every
#                                    laser-compare concern in one file
#                                    so the smoke test's
#                                    `subprocess.run([sys.executable,
#                                    "-m", "alayalite.bench.laser_compare",
#                                    ...])` invocation has a single
#                                    import surface; splitting would
#                                    not reduce overall complexity.
# pylint: disable=import-error  # `numpy` is a hard runtime dependency
#                                  declared in `pyproject.toml`. Pylint
#                                  invoked through `uv tool run` cannot
#                                  resolve the editable install's
#                                  third-party imports; the import
#                                  succeeds at run time and the smoke
#                                  test exercises the path end-to-end.
# pylint: disable=too-many-instance-attributes  # `LaserPrefix` carries
#                                                 the full discovery
#                                                 record (src_dir,
#                                                 basename, prefix_path,
#                                                 R, MD, N, filename_R,
#                                                 filename_MD); each
#                                                 field is consumed by
#                                                 a different stage.

from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ._datasets import DatasetSpec, load_laser_files
from ._engines import probe_disk_laser_supported
from ._metrics import (
    SCHEMA_VERSION,
    latency_summary_us,
    peak_rss_kb_and_unit,
    recall_at_k,
    render_raw_json,
    segment_bytes,
    segment_count,
    write_json,
)
from ._provenance import collect_provenance

# v1 binding contract: `DiskCollection.import_laser_segment` Python signature
# is `(src_dir, labels, *, copy=True)` — it does NOT expose
# `LaserSegmentImportParams.search_dram_budget_gb` (whose C++ default is
# 0.5F). To keep both paths matched on the DRAM side per design D7, the
# harness rejects any non-default `--dram-budget-gb` rather than silently
# use a mismatched budget on one side. See `tasks.md` Section 1, finding 1.1.
_DEFAULT_DRAM_BUDGET_GB = 0.5
_DRAM_BUDGET_EQ_EPS = 1e-9

# v1 binding contract: `LaserSegmentSearcher::search`
# (`include/index/disk/laser_segment_searcher.hpp:387`) hard-codes
# `num_threads=1`; `DiskSearchOptions` has no `num_threads` field. The
# `DiskCollection.search` Python binding therefore cannot route
# `--num-threads N > 1` to the disk_laser searcher. By analogy with D7,
# the harness rejects `--num-threads != 1` rather than running an
# asymmetric (native multi-threaded vs disk_laser pinned-to-1) comparison.
# Spec scenario "Multi-thread requests are refused" mandates the
# refuse-at-CLI behaviour; the future "Multi-thread emits a noise
# warning" requirement (currently held in reserve in the spec body)
# becomes reachable once a follow-up binding change exposes
# `num_threads` through `DiskSearchOptions`. See `tasks.md` Section 1,
# finding 1.2.
_PINNED_NUM_THREADS = 1


# ──────────────────────────────────────────────────────────────────────────
# Fixture-prefix discovery (design D6).
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LaserPrefix:
    """Resolved LASER artifact prefix and constructor parameters.

    ``basename`` is the ``dsqg_<seg>_R<R>_MD<MD>`` filename root without the
    ``.index`` suffix. Three views are exposed:

    * ``prefix_path`` is the full ``<src_dir>/<basename>`` path that
      ``LaserSegmentImporter`` uses when copying / linking artifacts; it
      maps 1-to-1 onto the on-disk filename.
    * ``base_prefix_path`` is ``<src_dir>/dsqg_<seg>`` (with the trailing
      ``_R<filename_R>_MD<filename_MD>`` token stripped). The native
      ``Index.load()`` path expects this shape because
      ``QuantizedGraph::gen_index_path`` adds back the
      ``_R<R>_MD<dimension_>.index`` suffix using the runtime
      ``degree_bound_`` / ``dimension_`` (`qg.hpp:168-171`).
    * ``R`` / ``MD`` / ``N`` are *effective* values (after CLI overrides);
      ``filename_R`` / ``filename_MD`` track the values that were
      actually parsed from the on-disk filename so the strip in
      ``base_prefix_path`` always targets the real suffix even when the
      user passes ``--laser-R`` / ``--laser-MD`` to override.
    """

    src_dir: Path
    basename: str
    prefix_path: Path  # = src_dir / basename
    R: int
    MD: int
    N: int
    filename_R: int  # parsed from `_R<R>` token in `basename` (pre-override)
    filename_MD: int  # parsed from `_MD<MD>` token in `basename` (pre-override)

    @property
    def index_file(self) -> Path:
        """Full path to the ``.index`` artifact for this fixture."""
        return self.src_dir / f"{self.basename}.index"

    @property
    def base_prefix_path(self) -> Path:
        """Prefix accepted by native ``Index.load`` (sans ``_R<R>_MD<MD>``)."""
        suffix = f"_R{self.filename_R}_MD{self.filename_MD}"
        if not self.basename.endswith(suffix):
            raise ValueError(
                f"laser_compare: internal error — basename {self.basename!r} "
                f"does not end with expected suffix {suffix!r}; cannot "
                "derive base prefix for native Index.load"
            )
        return self.src_dir / self.basename[: -len(suffix)]


def _parse_R_MD_from_filename(name: str) -> tuple[int, int]:
    """Parse ``R`` and ``MD`` tokens from ``dsqg_<seg>_R<R>_MD<MD>.index``.

    Raises ``ValueError`` with the offending filename if either token is
    missing, non-integer, **or duplicated** (e.g. ``..._R64_R32_MD128``);
    silently picking the last match would mask malformed fixtures. The
    caller is expected to surface the message to the user verbatim.
    """
    if not name.endswith(".index"):
        raise ValueError(f"laser_compare: expected .index suffix on {name}")
    stem = name[: -len(".index")]
    parts = stem.split("_")
    R_values: list[int] = []
    MD_values: list[int] = []
    for token in parts:
        if token.startswith("MD") and len(token) > 2 and token[2:].isdigit():
            MD_values.append(int(token[2:]))
        elif token.startswith("R") and len(token) > 1 and token[1:].isdigit():
            R_values.append(int(token[1:]))
    if not R_values or not MD_values:
        raise ValueError(
            f"laser_compare: could not parse R / MD tokens from filename {name}; expected dsqg_<seg>_R<R>_MD<MD>.index"
        )
    if len(R_values) > 1 or len(MD_values) > 1:
        raise ValueError(
            "laser_compare: duplicate R / MD tokens in filename "
            f"{name} (R={R_values}, MD={MD_values}); expected exactly one "
            "of each in dsqg_<seg>_R<R>_MD<MD>.index"
        )
    return R_values[0], MD_values[0]


def _read_index_count(index_path: Path) -> int:
    """Read the leading qword of an ``.index`` file as little-endian uint64.

    Mirrors `LaserSegmentImporter`'s `read_index_count`
    (`include/index/disk/laser_segment_importer.hpp:178-207`): the C++
    side parses the first eight bytes the same way, so we MUST use the
    same `< Q` struct unpack to derive ``N``.
    """
    with index_path.open("rb") as f:
        head = f.read(8)
    if len(head) != 8:
        raise ValueError(f"laser_compare: {index_path} is shorter than 8 bytes; cannot read N from the leading qword")
    return struct.unpack("<Q", head)[0]


def discover_laser_prefix(
    src_dir: Path,
    *,
    R_override: Optional[int] = None,
    MD_override: Optional[int] = None,
    N_override: Optional[int] = None,
) -> LaserPrefix:
    """Find the single ``dsqg_*_R*_MD*.index`` prefix under ``src_dir``.

    Per design D6: the directory MUST contain exactly one match; multi-segment
    fixtures are out of scope. Filename-derived ``R``/``MD`` and
    file-header-derived ``N`` are returned, with optional CLI overrides
    that emit warnings to stderr (the override path is for diagnostic use).
    """
    if not src_dir.is_dir():
        raise ValueError(f"laser_compare: --laser-src-dir does not exist or is not a directory: {src_dir}")

    matches = sorted(src_dir.glob("dsqg_*_R*_MD*.index"))
    if not matches:
        raise ValueError(
            "laser_compare: no LASER artifact prefix found under "
            f"{src_dir}; expected exactly one file matching "
            "dsqg_*_R*_MD*.index"
        )
    if len(matches) > 1:
        listing = ", ".join(p.name for p in matches)
        raise ValueError(
            "laser_compare: multiple LASER artifact prefixes found under "
            f"{src_dir}; expected exactly one. Matches: {listing}"
        )

    index_path = matches[0]
    parsed_R, parsed_MD = _parse_R_MD_from_filename(index_path.name)
    parsed_N = _read_index_count(index_path)

    R = parsed_R
    MD = parsed_MD
    N = parsed_N
    if R_override is not None and R_override != parsed_R:
        print(
            f"laser_compare: --laser-R {R_override} overrides discovered value R={parsed_R} from {index_path.name}",
            file=sys.stderr,
        )
        R = R_override
    if MD_override is not None and MD_override != parsed_MD:
        print(
            f"laser_compare: --laser-MD {MD_override} overrides discovered value MD={parsed_MD} from {index_path.name}",
            file=sys.stderr,
        )
        MD = MD_override
    if N_override is not None and N_override != parsed_N:
        print(
            f"laser_compare: --laser-N {N_override} overrides discovered "
            f"value N={parsed_N} read from leading qword of {index_path.name}",
            file=sys.stderr,
        )
        N = N_override

    basename = index_path.stem
    return LaserPrefix(
        src_dir=src_dir,
        basename=basename,
        prefix_path=src_dir / basename,
        R=R,
        MD=MD,
        N=N,
        filename_R=parsed_R,
        filename_MD=parsed_MD,
    )


# ──────────────────────────────────────────────────────────────────────────
# argparse + post-validation.
# ──────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alayalite.bench.laser_compare",
        description=(
            "Paired LASER read-path benchmark: native alayalite.laser.Index vs DiskCollection(index_type='disk_laser')."
        ),
    )

    parser.add_argument("--laser-src-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)

    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=128)
    parser.add_argument("--beam-width", dest="beam_width", type=int, default=4)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=1)
    parser.add_argument(
        "--dram-budget-gb",
        dest="dram_budget_gb",
        type=float,
        default=_DEFAULT_DRAM_BUDGET_GB,
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--drop-page-cache", dest="drop_page_cache", action="store_true")

    # Diagnostic overrides; values different from discovery emit a warning.
    parser.add_argument("--laser-R", dest="laser_R", type=int, default=None)
    parser.add_argument("--laser-MD", dest="laser_MD", type=int, default=None)
    parser.add_argument("--laser-N", dest="laser_N", type=int, default=None)

    # NOTE: --metric is intentionally absent; v1 LASER is L2-only and the
    # spec forbids any flag that would let the two paths diverge.
    return parser


class _CliRefusal(SystemExit):
    """Raised by `_validate_args` so `main()` can exit non-zero with a
    one-line stderr message instead of an argparse usage banner."""


def _validate_args(args: argparse.Namespace) -> None:
    """Reject CLI combinations the v1 binding cannot honour.

    Both refusals here are derived from Stage 1 pre-flight findings
    (`tasks.md` Section 1):

    * ``--dram-budget-gb != 0.5`` — the disk_laser binding does not
      expose ``LaserSegmentImportParams.search_dram_budget_gb`` as a
      kwarg, so the disk_laser segment would fall back to the C++
      default of 0.5 while the native side honours the user value.
      Per design D7, refuse rather than silently use mismatched budgets.
    * ``--num-threads != 1`` — ``LaserSegmentSearcher::search`` hard-codes
      ``num_threads = 1`` and ``DiskSearchOptions`` has no ``num_threads``
      field, so the disk_laser side would always run single-threaded
      regardless of the flag. Refuse rather than run an asymmetric
      comparison; this matches spec scenario "Multi-thread requests
      are refused".
    """
    if abs(args.dram_budget_gb - _DEFAULT_DRAM_BUDGET_GB) > _DRAM_BUDGET_EQ_EPS:
        raise _CliRefusal(
            f"laser_compare: --dram-budget-gb={args.dram_budget_gb} is not "
            f"supported in v1 (DiskCollection.import_laser_segment Python "
            f"binding does not expose search_dram_budget_gb; default 0.5 "
            f"is the only value that keeps both paths matched). See OpenSpec "
            f"change python-laser-native-equivalence-benchmark, tasks.md "
            f"finding 1.1."
        )
    if args.num_threads != _PINNED_NUM_THREADS:
        raise _CliRefusal(
            f"laser_compare: --num-threads={args.num_threads} is not "
            f"supported in v1 (DiskCollection.search Python binding has no "
            f"num_threads parameter; LaserSegmentSearcher::search hard-codes "
            f"num_threads=1 at laser_segment_searcher.hpp:387). Only "
            f"--num-threads=1 produces a fair comparison. See OpenSpec "
            f"change python-laser-native-equivalence-benchmark, tasks.md "
            f"finding 1.2."
        )

    if args.k <= 0:
        raise _CliRefusal(f"laser_compare: --k must be > 0 (got {args.k})")
    if args.ef <= 0:
        raise _CliRefusal(f"laser_compare: --ef must be > 0 (got {args.ef})")
    if args.beam_width <= 0:
        raise _CliRefusal(f"laser_compare: --beam-width must be > 0 (got {args.beam_width})")
    if args.warmup < 0:
        raise _CliRefusal(f"laser_compare: --warmup must be >= 0 (got {args.warmup})")
    if args.queries <= 0:
        raise _CliRefusal(f"laser_compare: --queries must be > 0 (got {args.queries})")
    # Positive checks for sizing and override flags. The native LASER
    # `Index(...)` constructor requires `num_elements`, `main_dimension`,
    # `dimension`, `degree_bound` to all be strictly positive
    # (`include/index/graph/laser/qg/qg.hpp`). Catching this at argparse
    # time keeps the binding error from leaking through later.
    if args.n <= 0:
        raise _CliRefusal(f"laser_compare: --n must be > 0 (got {args.n})")
    if args.dim <= 0:
        raise _CliRefusal(f"laser_compare: --dim must be > 0 (got {args.dim})")
    for label, value in (
        ("--laser-R", args.laser_R),
        ("--laser-MD", args.laser_MD),
        ("--laser-N", args.laser_N),
    ):
        if value is not None and value <= 0:
            raise _CliRefusal(f"laser_compare: {label} must be > 0 if set (got {value})")


# ──────────────────────────────────────────────────────────────────────────
# Dataset loading.
# ──────────────────────────────────────────────────────────────────────────


def _load_dataset(args: argparse.Namespace, prefix: LaserPrefix) -> DatasetSpec:
    """Load the paired DatasetSpec via `_datasets.load_laser_files`.

    Per design D2: reuse the existing loader exactly so cross-harness diff
    tools see the same `dataset_sha256_prefix` for the same fixture, including
    the `--laser-src-dir` content hash that `_datasets._sha16_directory`
    folds in. The loader is invoked exactly once per run; both paths consume
    the resulting `DatasetSpec` in-process, so the resulting `sha16` is
    path-independent by construction (verifying "identical regardless of
    which path runs first" per task 2.4).
    """
    dataset = load_laser_files(
        n=args.n,
        dim=args.dim,
        query_count=args.queries,
        seed=args.seed,
        vectors_path=args.vectors,
        queries_path=args.queries_path,
        ground_truth_path=args.ground_truth,
        laser_src_dir=prefix.src_dir,
    )
    if dataset.dim != prefix.MD:
        raise _CliRefusal(
            f"laser_compare: dataset dim ({dataset.dim}) does not match "
            f"LASER fixture MD ({prefix.MD}); v1 LASER constraint per "
            f"LaserSegmentImporter requires main_dim == dim."
        )
    if dataset.n != prefix.N:
        raise _CliRefusal(
            f"laser_compare: dataset n ({dataset.n}) does not match "
            f"LASER fixture N ({prefix.N}) read from "
            f"{prefix.index_file.name} leading qword."
        )
    return dataset


# ──────────────────────────────────────────────────────────────────────────
# Native-path adapter (`alayalite.laser.Index`).
# ──────────────────────────────────────────────────────────────────────────


def _recall_block(predictions: list[list[int]], dataset: DatasetSpec, top_k: int) -> dict:
    """Compute the ``{recall_at_*, recall_at_top_k, recall_status}`` block.

    Mirrors `_engines._finish_results` for the fixed `recall_at_1/10/100`
    triplet so the per-path raws are shape-compatible with existing
    `bench_disk_laser` outputs. Adds a ``recall_at_top_k`` field equal to
    `recall_at_k(predictions, gt, top_k)` so spec D5's
    `recall_delta = disk_laser.recall_at_top_k - native.recall_at_top_k`
    formula has a single source of truth regardless of whether
    ``args.k`` is one of {1, 10, 100} or a custom value (e.g.
    ``--k 5``). Recall is computed in label space; the caller is
    responsible for translating PIDs first per design D3.
    """
    if dataset.ground_truth is None:
        return {
            "recall_at_1": None,
            "recall_at_10": None,
            "recall_at_100": None,
            "recall_at_top_k": None,
            "recall_status": "missing_ground_truth",
        }
    gt = dataset.ground_truth[: len(predictions)]
    fixed_depths = {f"recall_at_{depth}": recall_at_k(predictions, gt, depth) for depth in (1, 10, 100)}
    custom_recall = (
        fixed_depths[f"recall_at_{top_k}"] if top_k in (1, 10, 100) else recall_at_k(predictions, gt, int(top_k))
    )
    return {
        **fixed_depths,
        "recall_at_top_k": custom_recall,
        "recall_status": "computed",
    }


def _translate_pids_to_labels(pids: np.ndarray, dataset: DatasetSpec, *, query_index: int) -> list[int]:
    """Convert a uint32 PID array to external labels via ``dataset.labels``.

    Per design D3 + spec scenario "PID translation rejects out-of-range
    PIDs": raise immediately on any PID >= dataset.n with a message
    naming the bad PID, the query index, and dataset.n. Silent
    masking / wrap-around would undercount recall.
    """
    if pids.size == 0:
        return []
    pids_i = np.asarray(pids, dtype=np.int64)
    bad_mask = (pids_i < 0) | (pids_i >= dataset.n)
    if bool(bad_mask.any()):
        bad_pid = int(pids_i[bad_mask][0])
        raise RuntimeError(
            f"laser_compare: native search returned PID={bad_pid} for "
            f"query index {query_index}, but dataset.n={dataset.n}; "
            "refusing to silently mask before recall computation."
        )
    return [int(label) for label in dataset.labels[pids_i]]


def _run_native_path(dataset: DatasetSpec, params: dict, prefix: LaserPrefix) -> dict:
    """Execute the native ``alayalite.laser.Index`` path: load + warmup +
    timed loop, returning a result dict shaped like the existing
    ``bench_disk_laser`` raws.

    Per design D8 + Stage 1 finding 1.2: ``num_threads`` is pinned to 1
    here. The harness's CLI validation rejects any other value; this
    function trusts ``params['num_threads'] == 1``.

    Per non-goals: uses ``index.search(query, k)`` per-query, NOT
    ``batch_search`` — the disk_laser path is single-query only, and the
    paired comparison MUST issue identical loop shapes.
    """
    # pylint: disable=import-outside-toplevel
    # Lazy import: `alayalite.laser` is only loadable on builds with
    # `ALAYA_ENABLE_LASER=ON`. Stage 7 short-circuits on unsupported
    # builds before this function is reached, but the import is local
    # in case a caller invokes the function directly.
    from alayalite import laser as laser_module

    top_k = int(params["top_k"])
    ef = int(params["ef"])
    beam_width = int(params["beam_width"])
    num_threads = int(params["num_threads"])
    warmup = int(params["warmup"])
    timed_queries = int(params["queries"])
    dram_budget_gb = float(params["dram_budget_gb"])

    if dataset.queries.shape[0] < timed_queries:
        raise _CliRefusal(
            f"laser_compare: dataset has {dataset.queries.shape[0]} queries, fewer than --queries {timed_queries}"
        )

    build_start = time.perf_counter()
    index = laser_module.Index.from_prefix(
        str(prefix.base_prefix_path),
        dram_budget_gb=dram_budget_gb,
    )
    build_wall_s = time.perf_counter() - build_start
    index.set_params(ef_search=ef, num_threads=num_threads, beam_width=beam_width)

    for i in range(warmup):
        query = np.ascontiguousarray(dataset.queries[i % timed_queries], dtype=np.float32)
        index.search(query, top_k)

    latencies_us: list[float] = []
    timed_predictions: list[list[int]] = []
    loop_start = time.perf_counter()
    for i in range(timed_queries):
        query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
        t0 = time.perf_counter()
        pids = index.search(query, top_k)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        timed_predictions.append(_translate_pids_to_labels(pids, dataset, query_index=i))
    elapsed = time.perf_counter() - loop_start

    # Recall depth follows `_engines._measure_search`: do an extra
    # untimed pass at recall_depth when top_k < recall_depth so the
    # recall@10/@100 numbers stay comparable to the existing
    # `bench_disk_laser` raws. Otherwise reuse the timed predictions.
    recall_depth = min(100, dataset.n)
    if top_k >= recall_depth:
        recall_predictions = timed_predictions
    else:
        recall_predictions = []
        for i in range(timed_queries):
            query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
            pids = index.search(query, recall_depth)
            recall_predictions.append(_translate_pids_to_labels(pids, dataset, query_index=i))

    rss_kb, rss_unit = peak_rss_kb_and_unit()
    # Mirror `_engines._finish_results` shape (no `reason` key — that
    # field is added by `render_raw_json` only when the run is
    # `skipped` / `error` / similar) and add `recall_at_top_k` so the
    # comparison block can use a single field name for any `--k`.
    return {
        **_recall_block(recall_predictions, dataset, top_k),
        "qps": float(timed_queries / elapsed) if elapsed > 0 else 0.0,
        "latency_us": latency_summary_us(latencies_us),
        "build_wall_s": float(build_wall_s),
        "on_disk_bytes": None,
        "peak_rss_kb": int(rss_kb),
        "peak_rss_unit": rss_unit,
        "segment_count": None,
    }


# ──────────────────────────────────────────────────────────────────────────
# DiskCollection-path adapter (`alayalite.DiskCollection(disk_laser)`).
# ──────────────────────────────────────────────────────────────────────────


def _read_segment_manifest(seg_dir: Path) -> dict[str, str]:
    """Parse a v1 segment ``manifest.txt`` into a flat ``key=value`` dict.

    The C++ side serialises the manifest as plain ``key=value\\n`` lines
    (`include/index/disk/segment_manifest.hpp:287-313`); the importer's
    `x_laser_*` extras live in the same file. We only need a one-shot
    read for the DRAM-budget cross-check, so a hand-rolled parser keeps
    us out of any optional toml / yaml dependency.
    """
    text = (seg_dir / "manifest.txt").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _verify_manifest_dram_budget(col_path: Path, expected_gb: float) -> None:
    """Assert ``x_laser_search_dram_budget_gb`` matches expectation.

    Per spec scenario "DRAM budgets are equal on both sides" the harness
    must read the manifest and refuse if the value does not match. In
    v1 we always pass the C++ default 0.5 (binding does not expose the
    kwarg — Stage 1 finding 1.1), so this check protects against a
    future binding change silently routing a different default.
    """
    seg_root = col_path / "segments"
    seg_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir() and p.name.startswith("seg_"))
    if len(seg_dirs) != 1:
        raise RuntimeError(
            f"laser_compare: expected exactly one seg_* directory under {seg_root}, found {len(seg_dirs)}"
        )
    manifest = _read_segment_manifest(seg_dirs[0])
    raw = manifest.get("x_laser_search_dram_budget_gb")
    if raw is None:
        raise RuntimeError(
            "laser_compare: segment manifest is missing "
            f"x_laser_search_dram_budget_gb at {seg_dirs[0] / 'manifest.txt'}"
        )
    actual_gb = float(raw)
    if abs(actual_gb - expected_gb) > _DRAM_BUDGET_EQ_EPS:
        raise RuntimeError(
            f"laser_compare: disk_laser segment manifest reports "
            f"x_laser_search_dram_budget_gb={raw} but harness expected "
            f"{expected_gb}; refusing to run with mismatched budgets."
        )


def _run_disk_laser_path(dataset: DatasetSpec, params: dict, prefix: LaserPrefix, scratch_root: Path) -> dict:
    """Execute the ``DiskCollection(disk_laser)`` path: import + warmup +
    timed loop.

    Per design D7 + Stage 1 finding 1.1: ``import_laser_segment`` does
    not expose ``search_dram_budget_gb`` in the v1 binding. The harness
    only honours ``--dram-budget-gb 0.5`` (the C++ default) and
    cross-checks the resulting segment manifest before the timed loop
    starts.

    The ``DiskCollection`` and its scratch tree live entirely under
    ``scratch_root`` so the source ``--laser-src-dir`` is read-only
    (spec scenario "Source directory contents unchanged after run").

    Contract on ``scratch_root``: the caller owns the lifecycle. Each
    invocation MUST be passed a fresh, unique directory (the function
    creates ``<scratch_root>/coll`` and refuses to overwrite an existing
    DiskCollection). The caller is responsible for ``shutil.rmtree`` /
    ``TemporaryDirectory`` cleanup; this matches the
    ``disk_collection.py`` harness's per-run ``_scratch`` pattern
    rather than ``_engines.bench_disk_laser``'s in-function
    ``TemporaryDirectory`` because the paired harness's main loop
    needs the disk_laser fixture's cumulative on-disk size for
    `on_disk_bytes` reporting before tearing down.
    """
    # pylint: disable=import-outside-toplevel
    # Lazy import: see `_run_native_path` for the rationale (Stage 7
    # short-circuits unsupported builds before reaching this point).
    from alayalite import DiskCollection, MetricType

    top_k = int(params["top_k"])
    ef = int(params["ef"])
    beam_width = int(params["beam_width"])
    warmup = int(params["warmup"])
    timed_queries = int(params["queries"])
    expected_dram_gb = float(params["dram_budget_gb"])

    if dataset.queries.shape[0] < timed_queries:
        raise _CliRefusal(
            f"laser_compare: dataset has {dataset.queries.shape[0]} queries, fewer than --queries {timed_queries}"
        )

    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    col_path = scratch_root / "coll"

    col = DiskCollection(
        path=str(col_path),
        dim=int(dataset.dim),
        metric=MetricType.L2,
        index_type="disk_laser",
    )

    build_start = time.perf_counter()
    col.import_laser_segment(
        str(prefix.src_dir),
        np.ascontiguousarray(dataset.labels, dtype=np.uint64),
    )
    build_wall_s = time.perf_counter() - build_start

    _verify_manifest_dram_budget(col_path, expected_dram_gb)

    for i in range(warmup):
        query = np.ascontiguousarray(dataset.queries[i % timed_queries], dtype=np.float32)
        col.search(query, k=top_k, ef=ef, beam_width=beam_width)

    latencies_us: list[float] = []
    timed_predictions: list[list[int]] = []
    loop_start = time.perf_counter()
    for i in range(timed_queries):
        query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
        t0 = time.perf_counter()
        hits = col.search(query, k=top_k, ef=ef, beam_width=beam_width)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        # disk_laser already returns external uint64 labels (with
        # `distance = NaN` per `LaserSegmentSearcher::search` line 407);
        # no PID translation is needed on this side.
        timed_predictions.append([int(label) for label, _ in hits])
    elapsed = time.perf_counter() - loop_start

    recall_depth = min(100, dataset.n)
    if top_k >= recall_depth:
        recall_predictions = timed_predictions
    else:
        recall_predictions = []
        for i in range(timed_queries):
            query = np.ascontiguousarray(dataset.queries[i], dtype=np.float32)
            hits = col.search(query, k=recall_depth, ef=ef, beam_width=beam_width)
            recall_predictions.append([int(label) for label, _ in hits])

    rss_kb, rss_unit = peak_rss_kb_and_unit()
    return {
        **_recall_block(recall_predictions, dataset, top_k),
        "qps": float(timed_queries / elapsed) if elapsed > 0 else 0.0,
        "latency_us": latency_summary_us(latencies_us),
        "build_wall_s": float(build_wall_s),
        "on_disk_bytes": int(segment_bytes(col_path)),
        "peak_rss_kb": int(rss_kb),
        "peak_rss_unit": rss_unit,
        "segment_count": int(segment_count(col_path)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Page-cache drop helper (design D4 / spec "Drop page cache is opt-in").
# ──────────────────────────────────────────────────────────────────────────


class PageCacheDropError(RuntimeError):
    """Raised when the privileged page-cache drop fails for any reason.

    Caller policy is "fail loudly, never silently": the harness
    catches this in `main()` and re-emits as a non-zero exit with the
    failure reason — partial cache state is never accepted.
    """


def _drop_page_cache() -> None:
    """Drop the OS page cache between the two timed paths.

    Calls ``sudo /sbin/sysctl vm.drop_caches=3`` (matching the existing
    `_engines` convention) on Linux. On any failure (non-Linux,
    subprocess error, sudo refusal) this function raises
    ``PageCacheDropError`` so the caller can refuse to proceed with a
    partially-applied cache state per spec scenario "drop_caches
    failure refuses to run".
    """
    if sys.platform != "linux":
        raise PageCacheDropError(
            f"laser_compare: --drop-page-cache is only supported on Linux (current platform: {sys.platform})"
        )
    cmd = ["sudo", "/sbin/sysctl", "vm.drop_caches=3"]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PageCacheDropError(
            f"laser_compare: --drop-page-cache requires `sudo` and `/sbin/sysctl`; not found ({exc})"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PageCacheDropError(
            f"laser_compare: --drop-page-cache failed (exit {exc.returncode}): stderr={(exc.stderr or '').strip()!r}"
        ) from exc
    except OSError as exc:
        # Catches `PermissionError` (sudo binary present but not
        # exec-able), `BlockingIOError` from a paused tty prompt,
        # and any other OS-level failure that doesn't surface as
        # `FileNotFoundError` or `CalledProcessError`. The fail-loud
        # contract says we still refuse to proceed.
        raise PageCacheDropError(f"laser_compare: --drop-page-cache failed with OS error: {exc}") from exc
    # Defensive: subprocess.run(check=True) already raises on non-zero
    # but a future refactor that drops `check=True` would mask
    # failures; keep the explicit assertion.
    if completed.returncode != 0:
        raise PageCacheDropError(
            f"laser_compare: --drop-page-cache returned exit "
            f"{completed.returncode}: "
            f"stderr={(completed.stderr or '').strip()!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Comparison block + output writers (design D5 / D10).
# ──────────────────────────────────────────────────────────────────────────


_NUM_THREADS_WARNING_MSG = "adapter_overhead_pct may be dominated by scheduler noise at num_threads > 1"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    """Return ``numerator / denominator`` or ``None`` if the denominator
    is non-positive, ``None``, or non-finite (NaN / inf). The spec
    mandates JSON `null` (rather than `inf` / `nan` / a Python
    exception) for these zero / null cases — see scenario "comparison
    fields gracefully handle zero or null inputs".
    """
    if numerator is None or denominator is None:
        return None
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _safe_sub(left: float, right: float) -> Optional[float]:
    if left is None or right is None:
        return None
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return float(left) - float(right)


def _compute_comparison(native_raw: dict, disk_raw: dict, params: dict) -> dict:
    """Build the top-level ``comparison`` block per spec D5.

    Field formulas come straight from the spec table; null propagation
    is handled by ``_safe_div`` / ``_safe_sub`` so downstream tools can
    re-derive the four derived fields from the per-path raws and get
    the same value modulo IEEE-754 reordering.
    """
    n_results = native_raw["results"]
    d_results = disk_raw["results"]

    n_qps = n_results.get("qps")
    d_qps = d_results.get("qps")
    n_lat = n_results.get("latency_us") or {}
    d_lat = d_results.get("latency_us") or {}

    latency_us = {
        "native": {p: n_lat.get(p) for p in ("p50", "p95", "p99")},
        "disk_laser": {p: d_lat.get(p) for p in ("p50", "p95", "p99")},
    }
    latency_us_delta = {p: _safe_sub(d_lat.get(p), n_lat.get(p)) for p in ("p50", "p95", "p99")}

    # Both raws expose `recall_at_top_k` (computed at `args.k` regardless
    # of whether k is one of the fixed depths). The spec D5 formula uses
    # this single field name on both sides.
    n_recall = n_results.get("recall_at_top_k")
    d_recall = d_results.get("recall_at_top_k")
    recall_delta = _safe_sub(d_recall, n_recall)

    n_p50 = n_lat.get("p50")
    d_p50 = d_lat.get("p50")
    qps_ratio = _safe_div(d_qps, n_qps)
    adapter_overhead_pct = _safe_div(_safe_sub(d_p50, n_p50), n_p50)
    if adapter_overhead_pct is not None:
        adapter_overhead_pct = 100.0 * adapter_overhead_pct

    block = {
        "recall_delta": recall_delta,
        "qps_native": float(n_qps) if n_qps is not None else None,
        "qps_disk_laser": float(d_qps) if d_qps is not None else None,
        "qps_ratio": qps_ratio,
        "latency_us": latency_us,
        "latency_us_delta": latency_us_delta,
        "adapter_overhead_pct": adapter_overhead_pct,
    }

    # In v1 the harness rejects `--num-threads != 1` at validation time
    # (Stage 1 finding 1.2), so this branch is unreachable; we keep it
    # so the field appears verbatim once a follow-up binding change
    # exposes `num_threads` through `DiskSearchOptions`.
    if int(params.get("num_threads", 1)) > 1:
        block["num_threads_warning"] = _NUM_THREADS_WARNING_MSG

    if params.get("drop_page_cache"):
        block["page_cache"] = "dropped_between_paths"

    return block


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_qps(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def _format_lat(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _render_summary_md(summary: dict) -> str:
    """Render a human-readable comparison table for ``summary.md``.

    Mirrors `_metrics.render_summary_md`'s header style so the LASER
    paired summary looks at home alongside `disk_collection.py`'s
    output. The body is a single comparison table — the harness
    produces exactly two raws, so a per-engine sweep grid is not
    needed.
    """
    provenance = summary["provenance"]
    comparison = summary["comparison"]
    params = summary["params"]

    lines = [
        "# LASER Native vs DiskCollection(disk_laser) Benchmark Summary",
        "",
        f"- schema_version: {summary['schema_version']}",
        f"- run_id: {summary['run_id']}",
        f"- git_commit_sha: {provenance['git_commit_sha']}",
        f"- git_dirty: {provenance['git_dirty']}",
        f"- timestamp_iso8601: {provenance['timestamp_iso8601']}",
        f"- cpu_model: {provenance['cpu_model']}",
        f"- cpu_count: {provenance['cpu_count']}",
        f"- compiler_flags: {provenance['compiler_flags']}",
        "",
        "## Parameters",
        "",
        ("| top_k | ef | beam_width | num_threads | warmup | queries | seed | dram_budget_gb |"),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| "
            + " | ".join(
                _format_metric(params[k])
                for k in (
                    "top_k",
                    "ef",
                    "beam_width",
                    "num_threads",
                    "warmup",
                    "queries",
                    "seed",
                    "dram_budget_gb",
                )
            )
            + " |"
        ),
        "",
        "## Comparison",
        "",
        "| metric | native | disk_laser | delta |",
        "|---|---:|---:|---:|",
    ]
    n_lat = comparison["latency_us"]["native"]
    d_lat = comparison["latency_us"]["disk_laser"]
    delta_lat = comparison["latency_us_delta"]
    raws_by_engine = {raw["engine"]: raw["results"] for raw in summary["raws"]}
    n_recall = raws_by_engine.get("native_laser", {}).get("recall_at_top_k")
    d_recall = raws_by_engine.get("disk_laser", {}).get("recall_at_top_k")
    lines.extend(
        [
            (
                f"| qps | {_format_qps(comparison['qps_native'])} | "
                f"{_format_qps(comparison['qps_disk_laser'])} | "
                f"ratio={_format_ratio(comparison['qps_ratio'])} |"
            ),
            (
                f"| recall@{params['top_k']} | "
                f"{_format_metric(n_recall)} | {_format_metric(d_recall)} | "
                f"{_format_metric(comparison['recall_delta'])} |"
            ),
            (
                f"| latency_us.p50 | {_format_lat(n_lat['p50'])} | "
                f"{_format_lat(d_lat['p50'])} | "
                f"{_format_lat(delta_lat['p50'])} |"
            ),
            (
                f"| latency_us.p95 | {_format_lat(n_lat['p95'])} | "
                f"{_format_lat(d_lat['p95'])} | "
                f"{_format_lat(delta_lat['p95'])} |"
            ),
            (
                f"| latency_us.p99 | {_format_lat(n_lat['p99'])} | "
                f"{_format_lat(d_lat['p99'])} | "
                f"{_format_lat(delta_lat['p99'])} |"
            ),
            "",
            (
                "**adapter_overhead_pct** "
                f"(`100 * (disk_p50 - native_p50) / native_p50`): "
                f"{_format_ratio(comparison['adapter_overhead_pct'])}"
            ),
            "",
        ]
    )
    if "num_threads_warning" in comparison:
        lines.append(f"> {comparison['num_threads_warning']}")
        lines.append("")
    if "page_cache" in comparison:
        lines.append(f"> page_cache: {comparison['page_cache']}")
        lines.append("")
    return "\n".join(lines)


def _default_run_id() -> str:
    """Return ``<utc_microsecond_stamp>_<git_sha8>`` matching
    `disk_collection.py`'s convention so cross-harness diff tools see
    the same run-id shape.

    When ``git rev-parse`` is unavailable (no `.git` dir, missing git,
    etc.) the SHA slot is filled with eight zeros so the resulting id
    still matches the spec regex
    ``^[0-9]{8}T[0-9]{6}_[0-9]{6}_[0-9a-f]{8}$``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    sha = "00000000"
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if rev.returncode == 0:
            candidate = rev.stdout.strip()[:8]
            if len(candidate) == 8 and all(ch in "0123456789abcdef" for ch in candidate.lower()):
                sha = candidate
    except (FileNotFoundError, OSError):
        pass
    return f"{stamp}_{sha}"


def _allocate_run_dir(out_root: Path, run_id: str) -> Path:
    """Return ``out_root/run_id`` if free, else ``out_root/run_id_NNN``.

    Mirrors `disk_collection.py:_allocate_run_dir`'s collision
    strategy: suffix `_001` ... `_999`; raise if all 999 are taken.
    The spec mandates this layout in scenario "run_id collision falls
    back to suffixed names".
    """
    candidate = Path(out_root) / run_id
    if not (candidate / "summary.json").exists():
        return candidate
    for i in range(1, 1000):
        suffixed = Path(out_root) / f"{run_id}_{i:03d}"
        if not (suffixed / "summary.json").exists():
            return suffixed
    raise RuntimeError(f"laser_compare: could not find a free run dir for {run_id} after 999 attempts")


def _write_to_run_dir(
    run_dir: Path,
    summary: dict,
    summary_md: str,
    native_raw: dict,
    disk_raw: dict,
) -> None:
    """Write the per-run output tree into a pre-allocated ``run_dir``.

    Splitting allocation (`_allocate_run_dir`) from writing keeps the
    scratch tree anchored to the same `run_dir` even when collision
    handling shifted the directory to a `_NNN` suffix.
    """
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    native_name = f"{native_raw['engine']}_{native_raw['dataset']}_{native_raw['metric']}.json"
    disk_name = f"{disk_raw['engine']}_{disk_raw['dataset']}_{disk_raw['metric']}.json"
    write_json(raw_dir / native_name, native_raw)
    write_json(raw_dir / disk_name, disk_raw)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# main() — full pipeline.
# ──────────────────────────────────────────────────────────────────────────


def _params_dict(args: argparse.Namespace) -> dict:
    """Build the params dict consumed by both adapters and by
    `_compute_comparison` / `_render_summary_md`.

    Carries every ``(top_k, ef, beam_width, num_threads, queries,
    warmup, seed, dram_budget_gb)`` axis listed in the spec's "Paired
    CLI" requirement plus the `drop_page_cache` flag and the cli
    `metric` (always "L2" in v1).
    """
    return {
        "top_k": int(args.k),
        "ef": int(args.ef),
        "beam_width": int(args.beam_width),
        "num_threads": int(args.num_threads),
        "queries": int(args.queries),
        "warmup": int(args.warmup),
        "seed": int(args.seed),
        "dram_budget_gb": float(args.dram_budget_gb),
        "drop_page_cache": bool(args.drop_page_cache),
        "metric": "L2",
    }


def _build_summary(
    *,
    run_id: str,
    provenance: dict,
    params: dict,
    native_raw: dict,
    disk_raw: dict,
    comparison: dict,
) -> dict:
    """Assemble the top-level `summary.json` payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "provenance": provenance,
        "params": params,
        "comparison": comparison,
        "raws": [native_raw, disk_raw],
    }


_UNSUPPORTED_SKIP_LINE = "laser_compare: skipped (disk_laser not available on this build)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m alayalite.bench.laser_compare``.

    Returns the process exit code: 0 on success or unsupported-build
    skip; 2 on argparse / discovery / dataset validation failure;
    3 on ``--drop-page-cache`` subprocess failure. Argparse-level
    errors raise ``SystemExit`` directly via the parser, while CLI
    refusals raise ``_CliRefusal`` (a ``SystemExit`` subclass).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _validate_args(args)

    # Probe support before any functional LASER work happens. The probe
    # itself does construct a tiny `DiskCollection(index_type="disk_laser")`
    # — that's how it tells whether the engine is built (the binding
    # raises a "not implemented in v1" runtime error on unsupported
    # builds, which the probe catches). The probe's scratch dir lives
    # under `args.out` once we've created it; on unsupported builds we
    # fall through to the skip line WITHOUT touching `args.out`, so a
    # user invoking the harness with a path they cannot write to still
    # gets a clean exit-0 skip rather than an unrelated permission
    # error.
    out_root = Path(args.out)
    if not probe_disk_laser_supported():
        print(_UNSUPPORTED_SKIP_LINE)
        return 0
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        prefix = discover_laser_prefix(
            args.laser_src_dir,
            R_override=args.laser_R,
            MD_override=args.laser_MD,
            N_override=args.laser_N,
        )
        dataset = _load_dataset(args, prefix)
    except ValueError as exc:
        # `_CliRefusal` is a `SystemExit` subclass and bypasses this
        # handler unchanged. Library-level `ValueError`s (discovery,
        # dataset path checks) are converted to a clean stderr line +
        # exit 2.
        print(str(exc), file=sys.stderr)
        return 2

    params = _params_dict(args)
    harness_argv = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    run_id = args.run_id or _default_run_id()

    # Provenance is collected exactly once per run and embedded in both
    # raws + summary.json (spec requirement "Provenance SHALL be
    # collected once and shared by both raws").
    provenance = collect_provenance(args.seed, dataset.sha16, harness_argv)

    # `out_root` is already created above (Stage 7 probe writes its
    # tmp dir under `out_root` for cache locality with the run).
    # Allocate the run dir first so the scratch tree is anchored to the
    # actual run dir (suffix-collision could otherwise leave scratch
    # under `<run_id>/` while the run lands at `<run_id>_001/`).
    run_dir = _allocate_run_dir(out_root, run_id)
    scratch_root = run_dir / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    disk_scratch = scratch_root / "disk_laser"

    try:
        # Per design D4: order is `native warmup → native timed →
        # disk_laser warmup → disk_laser timed`.
        native_result = _run_native_path(dataset, params, prefix)

        if args.drop_page_cache:
            try:
                _drop_page_cache()
            except PageCacheDropError as exc:
                # Refuse to proceed with a partially-applied cache state
                # per spec scenario "drop_caches failure refuses to run".
                # No `summary.json` is written — the run dir tree we
                # already created is harmless without it; `_scratch` is
                # cleaned by the surrounding `finally`.
                print(str(exc), file=sys.stderr)
                return 3

        disk_result = _run_disk_laser_path(dataset, params, prefix, disk_scratch)

        raw_params = {
            "top_k": params["top_k"],
            "ef": params["ef"],
            "beam_width": params["beam_width"],
            "num_threads": params["num_threads"],
            "warmup": params["warmup"],
            "dram_budget_gb": params["dram_budget_gb"],
            "drop_page_cache": params["drop_page_cache"],
        }

        native_raw = render_raw_json(
            native_result,
            provenance,
            raw_params,
            engine="native_laser",
            dataset=dataset.name,
            metric="L2",
            n=dataset.n,
            dim=dataset.dim,
            queries=params["queries"],
            run_id=run_id,
            ignored_args=[],
        )
        disk_raw = render_raw_json(
            disk_result,
            provenance,
            raw_params,
            engine="disk_laser",
            dataset=dataset.name,
            metric="L2",
            n=dataset.n,
            dim=dataset.dim,
            queries=params["queries"],
            run_id=run_id,
            ignored_args=[],
        )

        comparison = _compute_comparison(native_raw, disk_raw, params)
        summary = _build_summary(
            run_id=run_id,
            provenance=provenance,
            params=params,
            native_raw=native_raw,
            disk_raw=disk_raw,
            comparison=comparison,
        )
        summary_md = _render_summary_md(summary)
        _write_to_run_dir(run_dir, summary, summary_md, native_raw, disk_raw)
    finally:
        # Always clean up scratch, even if either path raised mid-run.
        shutil.rmtree(scratch_root, ignore_errors=True)

    print(f"laser_compare: wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
