# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tier A byte-equality comparator for Laser-upstream alignment artifacts.

Compares existing port (AlayaLite) Laser artifacts against upstream
Laser artifacts. The old mode that invoked `examples/laser/main.py`
through `vamana/pca/medoid/index` was retired when that example became
a compact `Index.fit` wrapper.

Usage:
    uv run python scripts/laser_alignment/tier_a_byte_equal.py \\
        --port-config examples/laser/configs/synth_20k_768d_alayaP.toml \\
        --upstream-config /md1/huangliang/alaya-dev/Laser/reproduce/configs/synth_20k_768d_origP.toml \\
        --out-root /md1/huangliang/alaya-dev/build_graph/laser_alignment/tier_a_<YYYYMMDD>/synth_20k_768d/ \\
        --skip-run

Exit codes:
    0  PASS (possibly with pca_demotion_reason = "python_stack_version_skew")
   10  PCA fail (tolerance exceeded OR dim/raw_dim mismatch)
   11  medoid fail
   12  rotator fail
   13  dsqg fail
   14  rabitq fail (reserved — RaBitQ is deterministic by construction)
   20  harness error (missing config, dataset name mismatch, pipeline crash)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# tomllib is stdlib from 3.11; tomli backport covers 3.9/3.10. The
# library pin in pyproject.toml ensures `tomli` is present via the
# `laser` extras group on older interpreters.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


PCA_DEMOTION_EPS: float = 1e-6

EXIT_PASS = 0
EXIT_PCA_FAIL = 10
EXIT_MEDOID_FAIL = 11
EXIT_ROTATOR_FAIL = 12
EXIT_DSQG_FAIL = 13
EXIT_RABITQ_FAIL = 14  # reserved; RaBitQ audit was DETERMINISTIC
EXIT_HARNESS_ERR = 20


# ── Low-level helpers ─────────────────────────────────────────────────────


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pca_bin(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Parse the laser._pca.save_pca_params binary layout.

    File format (see Laser/src/laser/pca.py and
    AlayaLite/python/src/alayalite/laser/_pca.py::save_pca_params):
      uint64   dim                      (= n_components_ = main_dim)
      float32  mean[raw_dim]
      float32  components[dim, raw_dim] (row-major)

    raw_dim is not in the header but is derivable from file size:
      file_size = 8 + 4*raw_dim*(1 + dim)
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (dim,) = struct.unpack("<Q", f.read(8))
        if dim <= 0:
            raise ValueError(f"{path}: invalid dim in header: {dim}")
        remaining = size - 8
        if remaining % (4 * (1 + dim)) != 0:
            raise ValueError(
                f"{path}: size/dim inconsistent: size={size} dim={dim} remainder={remaining % (4 * (1 + dim))}"
            )
        raw_dim = remaining // (4 * (1 + dim))
        mean = np.frombuffer(f.read(4 * raw_dim), dtype=np.float32).copy()
        components = np.frombuffer(f.read(4 * dim * raw_dim), dtype=np.float32).copy().reshape(dim, raw_dim)
    return int(dim), int(raw_dim), mean, components


def load_toml(toml_path: Path) -> dict:
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


# ── Top-down bisection ────────────────────────────────────────────────────


def _both_exist(port_dir: Path, upstream_dir: Path, name: str) -> tuple[Path, Path] | None:
    p = port_dir / name
    u = upstream_dir / name
    return (p, u) if p.exists() and u.exists() else None


def _sha_record(stage: str, filename: str, port_path: Path, upstream_path: Path) -> dict:
    return {
        "stage": stage,
        "file": filename,
        "port_sha": sha256_of(port_path),
        "upstream_sha": sha256_of(upstream_path),
    }


def bisection_compare(
    port_dir: Path,
    upstream_dir: Path,
    name: str,
    degree: int,
    main_dim: int,
) -> dict:
    """Top-down bisection over the build artifacts.

    Order: pca → medoids_indices → medoids → rotator_signs → dsqg.index.
    Stops at first divergence; reports named artifact + drift hypothesis.
    """
    result: dict = {"artifacts": []}

    pca_file = f"dsqg_{name}_pca.bin"
    medoid_idx_file = f"dsqg_{name}_medoids_indices"
    medoid_vec_file = f"dsqg_{name}_medoids"
    rotator_file = f"dsqg_{name}_rotator_signs.bin"
    dsqg_file = f"dsqg_{name}_R{degree}_MD{main_dim}.index"

    # Stage 1: PCA — byte-equal OR demotion path (element-wise |Δ| < 1e-6).
    paths = _both_exist(port_dir, upstream_dir, pca_file)
    if paths is None:
        result["status"] = "FAIL"
        result["stage"] = "pca"
        result["exit_code"] = EXIT_PCA_FAIL
        result["detail"] = (
            f"{pca_file} missing: port_exists={(port_dir / pca_file).exists()}, "
            f"upstream_exists={(upstream_dir / pca_file).exists()}"
        )
        return result
    port_pca, up_pca = paths
    pca_rec = _sha_record("pca", pca_file, port_pca, up_pca)
    if pca_rec["port_sha"] == pca_rec["upstream_sha"]:
        pca_rec["status"] = "PASS"
    else:
        try:
            p_dim, p_raw, p_mean, p_comp = parse_pca_bin(port_pca)
            u_dim, u_raw, u_mean, u_comp = parse_pca_bin(up_pca)
        except Exception as e:  # noqa: BLE001
            pca_rec["status"] = "ERROR"
            pca_rec["error"] = str(e)
            result["status"] = "FAIL"
            result["stage"] = "pca"
            result["exit_code"] = EXIT_PCA_FAIL
            result["detail"] = f"PCA parse failure: {e}"
            result["artifacts"].append(pca_rec)
            return result
        if (p_dim, p_raw) != (u_dim, u_raw):
            pca_rec["status"] = "FAIL"
            pca_rec["detail"] = (
                f"shape mismatch: port=(dim={p_dim}, raw_dim={p_raw}), upstream=(dim={u_dim}, raw_dim={u_raw})"
            )
            result["status"] = "FAIL"
            result["stage"] = "pca"
            result["exit_code"] = EXIT_PCA_FAIL
            result["detail"] = pca_rec["detail"]
            result["artifacts"].append(pca_rec)
            return result
        mean_delta = float(np.max(np.abs(p_mean - u_mean)))
        comp_delta = float(np.max(np.abs(p_comp - u_comp)))
        pca_rec["mean_delta"] = mean_delta
        pca_rec["comp_delta"] = comp_delta
        if mean_delta < PCA_DEMOTION_EPS and comp_delta < PCA_DEMOTION_EPS:
            pca_rec["status"] = "PASS_DEMOTED"
            pca_rec["pca_demotion_reason"] = "python_stack_version_skew"
        else:
            pca_rec["status"] = "FAIL"
            pca_rec["detail"] = (
                f"PCA tolerance exceeded: mean_delta={mean_delta:.3e}, "
                f"comp_delta={comp_delta:.3e}, eps={PCA_DEMOTION_EPS:.0e}"
            )
            result["status"] = "FAIL"
            result["stage"] = "pca"
            result["exit_code"] = EXIT_PCA_FAIL
            result["detail"] = pca_rec["detail"]
            result["artifacts"].append(pca_rec)
            return result
    result["artifacts"].append(pca_rec)

    # Stage 2: medoid (both indices + vectors).
    for mf in (medoid_idx_file, medoid_vec_file):
        paths = _both_exist(port_dir, upstream_dir, mf)
        if paths is None:
            result["status"] = "FAIL"
            result["stage"] = "medoid"
            result["exit_code"] = EXIT_MEDOID_FAIL
            result["detail"] = f"{mf} missing: port={(port_dir / mf).exists()}, upstream={(upstream_dir / mf).exists()}"
            return result
        rec = _sha_record("medoid", mf, *paths)
        if rec["port_sha"] != rec["upstream_sha"]:
            rec["status"] = "FAIL"
            result["status"] = "FAIL"
            result["stage"] = "medoid"
            result["exit_code"] = EXIT_MEDOID_FAIL
            result["detail"] = (
                f"medoid diverged ({mf}) — PCA was byte-equal, "
                f"so the bug lives in medoid selection (sample pick / "
                f"faiss k-means seed) or its input."
            )
            result["artifacts"].append(rec)
            return result
        rec["status"] = "PASS"
        result["artifacts"].append(rec)

    # Stage 3: rotator_signs.
    paths = _both_exist(port_dir, upstream_dir, rotator_file)
    if paths is None:
        result["status"] = "FAIL"
        result["stage"] = "rotator"
        result["exit_code"] = EXIT_ROTATOR_FAIL
        result["detail"] = (
            f"{rotator_file} missing: port={(port_dir / rotator_file).exists()}, "
            f"upstream={(upstream_dir / rotator_file).exists()} "
            f"(ensure the existing artifact trees include rotator_signs dumps)"
        )
        return result
    rec = _sha_record("rotator", rotator_file, *paths)
    if rec["port_sha"] != rec["upstream_sha"]:
        rec["status"] = "FAIL"
        result["status"] = "FAIL"
        result["stage"] = "rotator"
        result["exit_code"] = EXIT_ROTATOR_FAIL
        result["detail"] = (
            "FHTRotator RNG drift — rotator_signs.bin diverged despite "
            "matching PCA and medoid. Check the rotator generation settings "
            "used to produce both artifact trees."
        )
        result["artifacts"].append(rec)
        return result
    rec["status"] = "PASS"
    result["artifacts"].append(rec)

    # Stage 4: dsqg.index.
    paths = _both_exist(port_dir, upstream_dir, dsqg_file)
    if paths is None:
        result["status"] = "FAIL"
        result["stage"] = "dsqg"
        result["exit_code"] = EXIT_DSQG_FAIL
        result["detail"] = (
            f"{dsqg_file} missing: port={(port_dir / dsqg_file).exists()}, "
            f"upstream={(upstream_dir / dsqg_file).exists()}"
        )
        return result
    rec = _sha_record("dsqg", dsqg_file, *paths)
    if rec["port_sha"] != rec["upstream_sha"]:
        rec["status"] = "FAIL"
        result["status"] = "FAIL"
        result["stage"] = "dsqg"
        result["exit_code"] = EXIT_DSQG_FAIL
        result["detail"] = (
            "QG build drift — dsqg.index diverged despite all prior "
            "artifacts matching. The bug is in the RaBitQ packing / QG "
            "link / write_on_disk path (all RNG inputs were byte-equal)."
        )
        result["artifacts"].append(rec)
        return result
    rec["status"] = "PASS"
    result["artifacts"].append(rec)

    result["status"] = "PASS"
    result["exit_code"] = EXIT_PASS
    return result


# ── Report rendering ──────────────────────────────────────────────────────


def print_report(result: dict) -> None:
    print("\n========== Tier A Report ==========")
    print(f"Dataset:  {result.get('name', '?')}")
    print(f"Status:   {result.get('status', '?')}")
    if result.get("status") != "PASS":
        print(f"Stage:    {result.get('stage', '?')}")
        if result.get("detail"):
            print(f"Detail:   {result['detail']}")
    for art in result.get("artifacts", []):
        status = art.get("status", "?")
        stage = art.get("stage", "?")
        file_ = art.get("file", "")
        extra = ""
        if status == "PASS_DEMOTED":
            mean_d = art.get("mean_delta", float("nan"))
            comp_d = art.get("comp_delta", float("nan"))
            extra = f" (demoted: {art.get('pca_demotion_reason')}; mean_Δ={mean_d:.3e}, comp_Δ={comp_d:.3e})"
        elif status == "FAIL" and art.get("detail"):
            extra = f" ({art['detail']})"
        print(f"  [{status:>12}] {stage:>8}  {file_}{extra}")
    print("===================================\n")


# ── Entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--port-config",
        type=Path,
        required=True,
        help="AlayaLite alignment-mode TOML (examples/laser/configs/*_alayaP.toml)",
    )
    p.add_argument(
        "--upstream-config",
        type=Path,
        required=True,
        help="Laser alignment-mode TOML (Laser/reproduce/configs/*_origP.toml)",
    )
    p.add_argument("--out-root", type=Path, required=True, help="Root dir for per-side outputs + JSON diff report")
    p.add_argument(
        "--report", type=Path, default=None, help="JSON report path (default: <out-root>/tier_a_report.json)"
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Required. Compare existing <out-root>/{port,upstream} artifacts; pipeline invocation is retired.",
    )
    args = p.parse_args(argv)

    if not args.port_config.exists():
        print(f"[tier-a][err] port config missing: {args.port_config}", file=sys.stderr)
        return EXIT_HARNESS_ERR
    if not args.upstream_config.exists():
        print(f"[tier-a][err] upstream config missing: {args.upstream_config}", file=sys.stderr)
        return EXIT_HARNESS_ERR

    args.out_root.mkdir(parents=True, exist_ok=True)
    port_out = args.out_root / "port"
    upstream_out = args.out_root / "upstream"
    port_out.mkdir(parents=True, exist_ok=True)
    upstream_out.mkdir(parents=True, exist_ok=True)

    port_cfg = load_toml(args.port_config)
    up_cfg = load_toml(args.upstream_config)
    # Cross-side config consistency: all fields that flow into artifact
    # naming or physical layout MUST match between port and upstream,
    # otherwise the gate degenerates into a shape-mismatch failure at the
    # dsqg stage rather than a true build-drift failure. Fail fast with a
    # named harness error instead.
    for field in ("name", "degree", "main_dimension", "metric"):
        port_v = port_cfg["dataset"].get(field)
        up_v = up_cfg["dataset"].get(field)
        if port_v != up_v:
            print(
                f"[tier-a][err] dataset.{field} mismatch: port={port_v!r}, upstream={up_v!r}",
                file=sys.stderr,
            )
            return EXIT_HARNESS_ERR
    name = port_cfg["dataset"]["name"]
    degree = int(port_cfg["dataset"]["degree"])
    main_dim = int(port_cfg["dataset"]["main_dimension"])

    if not args.skip_run:
        print(
            "[tier-a][err] automatic pipeline invocation was retired when "
            "examples/laser/main.py became an Index.fit wrapper; pass "
            "--skip-run to compare existing artifacts.",
            file=sys.stderr,
        )
        return EXIT_HARNESS_ERR

    port_data_dir = port_out / "data" / name
    upstream_data_dir = upstream_out / "data" / name

    result = bisection_compare(port_data_dir, upstream_data_dir, name, degree, main_dim)
    result["name"] = name
    result["degree"] = degree
    result["main_dim"] = main_dim
    result["port_dir"] = str(port_data_dir)
    result["upstream_dir"] = str(upstream_data_dir)
    result["port_config"] = str(args.port_config)
    result["upstream_config"] = str(args.upstream_config)

    report_path = args.report or (args.out_root / "tier_a_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print_report(result)
    print(f"JSON report: {report_path}")

    return int(result.get("exit_code", EXIT_HARNESS_ERR))


if __name__ == "__main__":
    sys.exit(main())
