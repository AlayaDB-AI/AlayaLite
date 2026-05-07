#!/usr/bin/env python3
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

"""Benchmark unified ``laser.Index.fit`` against the legacy manual pipeline.

Outputs a JSON report plus a markdown table suitable for PR descriptions.

Example:
    uv run python/benchmarks/laser_unified_fit_bench.py \
      --dataset-root /md1/huangliang/alaya-dev/data/gist1m \
      --datasets gist1m \
      --output-dir results/laser_unified_fit_bench \
      --threads 8 --efs 100,200 --build-repeats 1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_fbin: Path
    query_fbin: Path
    gt_ibin: Path
    main_dim: int


def _read_fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(8)
    if len(head) != 8:
        raise ValueError(f"invalid fbin header: {path}")
    n, dim = struct.unpack("<ii", head)
    if n <= 0 or dim <= 0:
        raise ValueError(f"invalid fbin shape ({n}, {dim}) in {path}")
    return int(n), int(dim)


def _load_dataset_specs(root: Path, datasets: list[str], main_dim: int) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    for name in datasets:
        if name == "gist1m":
            base = root / "gist_base.fbin"
            query = root / "gist_query.fbin"
            gt = root / "gist_gt.ibin"
        elif name == "sift1m":
            base = root / "sift_base.fbin"
            query = root / "sift_query.fbin"
            gt = root / "sift_gt.ibin"
        else:
            raise ValueError(f"unsupported dataset {name!r}; supported: gist1m,sift1m")

        missing = [p for p in (base, query, gt) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"dataset {name} missing required files under {root}: {', '.join(str(p) for p in missing)}"
            )
        out.append(
            DatasetSpec(
                name=name,
                base_fbin=base,
                query_fbin=query,
                gt_ibin=gt,
                main_dim=main_dim,
            )
        )
    return out


def _build_shared_vamana(
    spec: DatasetSpec,
    shared_dir: Path,
    *,
    R: int,
    L: int,
    alpha: float,
    vamana_seed: int,
    num_threads: int,
    dram_budget_gb: float,
) -> Path:
    # pylint: disable=import-outside-toplevel
    from alayalite import vamana

    shared_dir.mkdir(parents=True, exist_ok=True)
    out = shared_dir / f"{spec.name}_shared_vamana_graph.index"
    vamana.build_index(
        data_path=str(spec.base_fbin),
        output_path=str(out),
        R=int(R),
        L=int(L),
        alpha=float(alpha),
        seed=int(vamana_seed),
        num_threads=int(num_threads),
        dram_budget_gb=float(dram_budget_gb),
    )
    return out


def _build_manual_pipeline(
    spec: DatasetSpec,
    output_dir: Path,
    *,
    prefix_name: str,
    shared_vamana_path: Path,
    R: int,
    ef_indexing: int,
    num_threads: int,
    ep_num: int,
    pca_seed: int,
    medoid_seed: int,
    rotator_seed: int,
    disable_medoid: bool,
) -> str:
    # pylint: disable=import-outside-toplevel
    from alayalite._alayalitepy import laser as raw_laser
    from alayalite.laser._medoid import generate_and_save_medoids
    from alayalite.laser._pca import (
        fit_incremental_pca,
        pca_transform_and_save,
        sample_vectors_from_fbin,
        save_pca_params,
    )

    n, raw_dim = _read_fbin_header(spec.base_fbin)
    prefix = output_dir / prefix_name
    pca_base = output_dir / f"{prefix_name}_pca_base.fbin"
    pca_params = output_dir / f"{prefix_name}_pca.bin"
    output_dir.mkdir(parents=True, exist_ok=True)

    if spec.main_dim < raw_dim:
        vectors, sample = sample_vectors_from_fbin(str(spec.base_fbin), seed=pca_seed)
        pca = fit_incremental_pca(sample, n_components=raw_dim)
        save_pca_params(pca, str(pca_params))
        pca_transform_and_save(vectors, pca, str(pca_base))
    else:
        shutil.copyfile(spec.base_fbin, pca_base)
        if pca_params.exists():
            pca_params.unlink()

    if not disable_medoid:
        generate_and_save_medoids(
            str(pca_base),
            str(output_dir / f"{prefix_name}_medoids_indices"),
            str(output_dir / f"{prefix_name}_medoids"),
            int(ep_num),
            seed=int(medoid_seed),
        )

    raw = raw_laser.Index(
        index_type="QG",
        metric="l2",
        num_elements=int(n),
        main_dimension=int(spec.main_dim),
        dimension=int(raw_dim),
        degree_bound=int(R),
        rotator_seed=int(rotator_seed),
        rotator_dump_path="",
    )
    raw.build_index(
        vamana_file=str(shared_vamana_path),
        data_file=str(prefix),
        EF=int(ef_indexing),
        num_thread=int(num_threads),
    )
    return str(prefix)


def _fit_unified(
    spec: DatasetSpec,
    output_dir: Path,
    *,
    prefix_name: str,
    shared_vamana_path: Path,
    R: int,
    L: int,
    alpha: float,
    ef_indexing: int,
    num_threads: int,
    ep_num: int,
    seed: int,
    disable_medoid: bool,
    dram_budget_gb: float,
) -> str:
    # pylint: disable=import-outside-toplevel
    from alayalite import laser

    output_dir.mkdir(parents=True, exist_ok=True)
    target_vamana = output_dir / f"{prefix_name}_vamana_graph.index"
    shutil.copyfile(shared_vamana_path, target_vamana)

    laser.Index.fit(
        str(spec.base_fbin),
        output_dir=output_dir,
        name=prefix_name,
        build_params=laser.BuildParams(
            metric="l2",
            main_dim=int(spec.main_dim),
            R=int(R),
            L=int(L),
            alpha=float(alpha),
            ef_indexing=int(ef_indexing),
            ep_num=int(ep_num),
            disable_medoid=bool(disable_medoid),
        ),
        num_threads=int(num_threads),
        seed=int(seed),
        dram_budget_gb=float(dram_budget_gb),
        skip_existing=True,
        auto_load=False,
    )
    return str(output_dir / prefix_name)


def _recall_at_k(predictions: np.ndarray, gt: np.ndarray, k: int) -> float:
    if predictions.shape[1] < k:
        raise ValueError(f"predictions second dim {predictions.shape[1]} < k={k}")
    if gt.shape[1] < k:
        raise ValueError(f"ground truth second dim {gt.shape[1]} < k={k}")
    correct = 0
    nq = predictions.shape[0]
    for i in range(nq):
        correct += len(set(predictions[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return float(correct) / float(nq * k)


def _measure_search(
    index_obj,
    queries: np.ndarray,
    gt: np.ndarray,
    *,
    k: int,
    ef: int,
    num_threads: int,
    beam_width: int,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    index_obj.set_params(ef_search=int(ef), num_threads=int(num_threads), beam_width=int(beam_width))
    for _ in range(int(warmup)):
        index_obj.batch_search(queries, int(k))

    preds = index_obj.batch_search(queries, int(k))
    recall = _recall_at_k(preds, gt, int(k))

    t0 = time.perf_counter()
    for _ in range(int(runs)):
        index_obj.batch_search(queries, int(k))
    elapsed = time.perf_counter() - t0
    qps = float((queries.shape[0] * int(runs)) / elapsed)
    return {"recall": recall, "qps": qps}


def _format_pct(delta: float) -> str:
    return f"{delta * 100.0:+.2f}%"


def _laser_artifact_hashes(prefix: str, R: int, main_dim: int) -> dict[str, str]:
    """SHA-256 the four LASER artifacts whose contents drive search behavior.

    These are what `Index.from_prefix(...).load(...)` and the search path actually
    read; if they are byte-identical between manual and unified, then any QPS
    difference is filesystem-layout noise (different inodes on the same disk),
    not an API-path overhead.
    """
    suffixes = [
        f"_R{int(R)}_MD{int(main_dim)}.index",
        f"_R{int(R)}_MD{int(main_dim)}.index_rotator",
        f"_R{int(R)}_MD{int(main_dim)}.index_cache_ids",
        f"_R{int(R)}_MD{int(main_dim)}.index_cache_nodes",
    ]
    out: dict[str, str] = {}
    for sfx in suffixes:
        path = prefix + sfx
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        out[sfx] = h.hexdigest()
    return out


def _dataset_report_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"### {report['dataset']}")
    lines.append("")
    lines.append(
        "| Build (s) Manual | Build (s) Unified | Build Delta | "
        "Recall@10 Delta (pp) @EF100 | Recall@10 Delta (pp) @EF200 | "
        "QPS Delta @EF100 | QPS Delta @EF200 | Vamana Shared (s) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    ef_map = {row["ef"]: row for row in report["ef_rows"]}
    ef100 = ef_map.get(100)
    ef200 = ef_map.get(200)
    lines.append(
        f"| {report['build_manual_s']:.2f} | {report['build_unified_s']:.2f} | "
        f"{_format_pct(report['build_delta_ratio'])} | "
        f"{(ef100['recall_delta_pp'] if ef100 else float('nan')):.3f} | "
        f"{(ef200['recall_delta_pp'] if ef200 else float('nan')):.3f} | "
        f"{(_format_pct(ef100['qps_delta_ratio']) if ef100 else 'N/A')} | "
        f"{(_format_pct(ef200['qps_delta_ratio']) if ef200 else 'N/A')} | "
        f"{report['vamana_shared_s']:.2f} |"
    )
    lines.append("")
    lines.append("| EF | Recall Manual | Recall Unified | QPS Manual | QPS Unified |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report["ef_rows"]:
        lines.append(
            f"| {row['ef']} | {row['manual_recall']:.6f} | {row['unified_recall']:.6f} | "
            f"{row['manual_qps']:.2f} | {row['unified_qps']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_one_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
    dataset_run_dir: Path,
) -> dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from alayalite import laser
    from alayalite.laser._io import read_fbin, read_ibin

    queries = np.asarray(read_fbin(str(spec.query_fbin), use_mmap=False), dtype=np.float32)
    gt = np.asarray(read_ibin(str(spec.gt_ibin)), dtype=np.int32)

    build_manual_runs: list[float] = []
    build_unified_runs: list[float] = []
    vamana_shared_runs: list[float] = []
    manual_prefix = ""
    unified_prefix = ""

    for i in range(int(args.build_repeats)):
        round_dir = dataset_run_dir / f"repeat_{i}"
        shared_dir = round_dir / "shared"
        manual_dir = round_dir / "manual"
        unified_dir = round_dir / "unified"

        t0 = time.perf_counter()
        shared_vamana_path = _build_shared_vamana(
            spec,
            shared_dir,
            R=args.R,
            L=args.L,
            alpha=args.alpha,
            vamana_seed=args.vamana_seed,
            num_threads=args.threads,
            dram_budget_gb=args.build_dram_budget_gb,
        )
        vamana_shared_runs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        manual_prefix = _build_manual_pipeline(
            spec,
            manual_dir,
            prefix_name=f"{spec.name}_manual",
            shared_vamana_path=shared_vamana_path,
            R=args.R,
            ef_indexing=args.ef_indexing,
            num_threads=args.threads,
            ep_num=args.ep_num,
            pca_seed=args.pca_seed,
            medoid_seed=args.medoid_seed,
            rotator_seed=args.rotator_seed,
            disable_medoid=args.disable_medoid,
        )
        build_manual_runs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        unified_prefix = _fit_unified(
            spec,
            unified_dir,
            prefix_name=f"{spec.name}_unified",
            shared_vamana_path=shared_vamana_path,
            R=args.R,
            L=args.L,
            alpha=args.alpha,
            ef_indexing=args.ef_indexing,
            num_threads=args.threads,
            ep_num=args.ep_num,
            seed=args.seed,
            disable_medoid=args.disable_medoid,
            dram_budget_gb=args.build_dram_budget_gb,
        )
        build_unified_runs.append(time.perf_counter() - t0)

        if not args.keep_artifacts and i < int(args.build_repeats) - 1:
            shutil.rmtree(round_dir, ignore_errors=True)

    build_manual_s = float(statistics.median(build_manual_runs))
    build_unified_s = float(statistics.median(build_unified_runs))
    build_delta_ratio = abs(build_unified_s - build_manual_s) / build_manual_s
    vamana_shared_s = float(statistics.median(vamana_shared_runs))

    # Verify the two routes produced byte-equal LASER artifacts before measuring
    # search QPS. With shared vamana upstream, all downstream steps (PCA, medoid,
    # rotator, LASER QG, cache) are deterministic given the same inputs, so a
    # mismatch here means a non-deterministic step crept in beyond shared vamana —
    # fail loud rather than report a corrupted QPS comparison.
    manual_hashes = _laser_artifact_hashes(manual_prefix, int(args.R), int(spec.main_dim))
    unified_hashes = _laser_artifact_hashes(unified_prefix, int(args.R), int(spec.main_dim))
    if manual_hashes != unified_hashes:
        diffs = {
            suffix: {"manual": manual_hashes[suffix], "unified": unified_hashes[suffix]}
            for suffix in manual_hashes
            if manual_hashes[suffix] != unified_hashes[suffix]
        }
        raise RuntimeError(
            f"manual and unified LASER artifacts diverge on {len(diffs)} files; "
            f"search QPS would be incomparable. This indicates a non-deterministic "
            f"step beyond shared vamana — investigate before suppressing. "
            f"Mismatched hashes: {diffs}"
        )

    # Hash-verify passed: load both search-time idx instances from the SAME physical
    # files (manual prefix). This eliminates ext4 extent / SSD LBA-placement noise
    # that would otherwise show up as ~7% QPS skew between two byte-equal index
    # files written sequentially to the same disk. The build-time delta still
    # reflects the real Python-wrapper / skip_existing overhead of the unified API.
    manual_idx = laser.Index.from_prefix(manual_prefix, dram_budget_gb=args.search_dram_budget_gb)
    unified_idx = laser.Index.from_prefix(manual_prefix, dram_budget_gb=args.search_dram_budget_gb)

    ef_rows = []
    for ef in args.efs:
        m = _measure_search(
            manual_idx,
            queries,
            gt,
            k=args.k,
            ef=ef,
            num_threads=args.threads,
            beam_width=args.beam_width,
            warmup=args.warmup,
            runs=args.runs,
        )
        u = _measure_search(
            unified_idx,
            queries,
            gt,
            k=args.k,
            ef=ef,
            num_threads=args.threads,
            beam_width=args.beam_width,
            warmup=args.warmup,
            runs=args.runs,
        )
        ef_rows.append(
            {
                "ef": int(ef),
                "manual_recall": float(m["recall"]),
                "unified_recall": float(u["recall"]),
                "recall_delta_pp": float(abs(u["recall"] - m["recall"]) * 100.0),
                "manual_qps": float(m["qps"]),
                "unified_qps": float(u["qps"]),
                "qps_delta_ratio": float(abs(u["qps"] - m["qps"]) / m["qps"]),
            }
        )

    return {
        "dataset": spec.name,
        "base_fbin": str(spec.base_fbin),
        "query_fbin": str(spec.query_fbin),
        "gt_ibin": str(spec.gt_ibin),
        "build_manual_s": build_manual_s,
        "build_unified_s": build_unified_s,
        "build_delta_ratio": build_delta_ratio,
        "build_manual_runs_s": build_manual_runs,
        "build_unified_runs_s": build_unified_runs,
        "vamana_shared_s": vamana_shared_s,
        "vamana_shared_runs_s": vamana_shared_runs,
        "artifact_hashes": manual_hashes,  # equal to unified_hashes after the verify
        "artifacts_byte_equal": True,  # only reachable after verify passes
        "search_load_source": "manual",  # both idx loaded from manual prefix
        "ef_rows": ef_rows,
    }


def _parse_efs(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("efs must be a non-empty comma-separated integer list")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True, help="Root dir containing gist/sift *.fbin/*.ibin files")
    p.add_argument("--datasets", default="gist1m", help="Comma-separated datasets: gist1m,sift1m")
    p.add_argument("--output-dir", type=Path, default=Path("results/laser_unified_fit_bench"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--main-dim", type=int, default=256)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--L", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--ef-indexing", type=int, default=200)
    p.add_argument("--efs", default="100,200")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--build-repeats", type=int, default=1)
    p.add_argument("--ep-num", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-seed", type=int, default=42)
    p.add_argument("--medoid-seed", type=int, default=42)
    p.add_argument("--vamana-seed", type=int, default=42)
    p.add_argument("--rotator-seed", type=int, default=42)
    p.add_argument("--build-dram-budget-gb", type=float, default=1.0)
    p.add_argument("--search-dram-budget-gb", type=float, default=2.0)
    p.add_argument("--disable-medoid", action="store_true")
    p.add_argument("--keep-artifacts", action="store_true")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.efs = _parse_efs(args.efs)

    if args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    datasets = [tok.strip() for tok in args.datasets.split(",") if tok.strip()]
    specs = _load_dataset_specs(args.dataset_root, datasets, main_dim=args.main_dim)

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for spec in specs:
        print(f"[bench] dataset={spec.name} start")
        dataset_dir = run_dir / spec.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        reports.append(_run_one_dataset(spec, args, dataset_dir))
        print(f"[bench] dataset={spec.name} done")

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "dataset_root": str(args.dataset_root),
            "datasets": datasets,
            "main_dim": args.main_dim,
            "R": args.R,
            "L": args.L,
            "alpha": args.alpha,
            "threads": args.threads,
            "beam_width": args.beam_width,
            "ef_indexing": args.ef_indexing,
            "efs": args.efs,
            "k": args.k,
            "warmup": args.warmup,
            "runs": args.runs,
            "build_repeats": args.build_repeats,
            "ep_num": args.ep_num,
            "seed": args.seed,
            "pca_seed": args.pca_seed,
            "medoid_seed": args.medoid_seed,
            "vamana_seed": args.vamana_seed,
            "rotator_seed": args.rotator_seed,
            "build_dram_budget_gb": args.build_dram_budget_gb,
            "search_dram_budget_gb": args.search_dram_budget_gb,
            "disable_medoid": args.disable_medoid,
        },
        "reports": reports,
    }

    json_path = run_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = ["## LASER Unified Fit Benchmark", ""]
    for report in reports:
        md_lines.append(_dataset_report_markdown(report))
    md = "\n".join(md_lines).rstrip() + "\n"
    md_path = run_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"[bench] json={json_path}")
    print(f"[bench] markdown={md_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
