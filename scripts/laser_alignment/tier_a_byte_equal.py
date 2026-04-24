"""Tier A byte-equality harness for the Laser-upstream alignment gate.

Compares the port (AlayaLite) Laser pipeline output against upstream
Laser pipeline output artifact-by-artifact under the Tier A invariants
(OMP_NUM_THREADS=1 env AND force_single_thread=true AND build_threads=1
AND AVX512, with all three seeds = 42). Reports PASS only when all
four canonical build artifacts SHA-256-match — or, for `dsqg_{name}_pca.bin`,
match under a documented Python-stack tolerance path (|Δ| < 1e-6).

Usage:
    uv run python scripts/laser_alignment/tier_a_byte_equal.py \\
        --port-config examples/laser/configs/synth_20k_768d_alayaP.toml \\
        --upstream-config /md1/huangliang/alaya-dev/Laser/reproduce/configs/synth_20k_768d_origP.toml \\
        --out-root /md1/huangliang/alaya-dev/build_graph/laser_alignment/tier_a_<YYYYMMDD>/synth_20k_768d/

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
import re
import struct
import subprocess
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

# Default repo locations for the gpu04 / bpettis-ubuntu research environment.
# Overridable via --port-repo / --upstream-repo CLI args for any other host.
ALAYA_REPO_DEFAULT = Path("/md1/huangliang/alaya-dev/AlayaLite")
LASER_REPO_DEFAULT = Path("/md1/huangliang/alaya-dev/Laser")

# Matches `output = "..."` exactly (not `output_dir`, `output2`, etc.).
# `^` is anchored to the stripped line after leading whitespace is removed
# in the scanner; we only need to match `output` as a full key token.
_OUTPUT_KEY_RE = re.compile(r"^output\s*=")

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
    """Parse the laser.pca.save_pca_params binary layout.

    File format (see Laser/src/laser/pca.py and
    AlayaLite/python/src/alayalite/laser/pca.py::save_pca_params):
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
                f"{path}: size/dim inconsistent: size={size} dim={dim} "
                f"remainder={remaining % (4 * (1 + dim))}"
            )
        raw_dim = remaining // (4 * (1 + dim))
        mean = np.frombuffer(f.read(4 * raw_dim), dtype=np.float32).copy()
        components = (
            np.frombuffer(f.read(4 * dim * raw_dim), dtype=np.float32)
            .copy()
            .reshape(dim, raw_dim)
        )
    return int(dim), int(raw_dim), mean, components


def rewrite_output(src_toml: Path, dst_toml: Path, new_output: str) -> None:
    """Copy src_toml to dst_toml, rewriting `[paths].output = "<new_output>"`.

    Line-level edit because tomllib has no writer and the Tier A configs
    have a stable `output = "…"` line under [paths]. Preserves comments
    and all other keys. Matches the `output` key strictly via regex so
    sibling keys like `output_dir` are not accidentally rewritten.
    Raises if zero or multiple matches are found (both indicate an
    unexpected config shape).
    """
    lines = src_toml.read_text().splitlines()
    new_lines: list[str] = []
    in_paths = False
    match_count = 0
    # TOML allows whitespace between `[` and the key, and a trailing space.
    # Normalise by comparing the stripped form to `[paths]`.
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_paths = stripped == "[paths]"
        # Skip comment lines; match only well-formed `output = ...`.
        if (
            in_paths
            and not stripped.startswith("#")
            and _OUTPUT_KEY_RE.match(stripped)
        ):
            new_lines.append(f'output = "{new_output}"')
            match_count += 1
        else:
            new_lines.append(line)
    if match_count == 0:
        raise RuntimeError(f"no [paths].output found in {src_toml}")
    if match_count > 1:
        raise RuntimeError(
            f"multiple [paths].output matches in {src_toml} (match_count={match_count}); "
            "refusing to rewrite an ambiguous config"
        )
    dst_toml.write_text("\n".join(new_lines) + "\n")


def load_toml(toml_path: Path) -> dict:
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


# ── Pipeline invocation ───────────────────────────────────────────────────

def run_pipeline(repo: Path, rel_entrypoint: str, toml_path: Path, steps: list[str]) -> None:
    """Invoke a Laser pipeline (port or upstream) via `uv run`.

    Sets OMP_NUM_THREADS=1 in the subprocess env. The TOML's
    force_single_thread/build_threads invariants are enforced by
    load_config inside the pipeline — this harness just guarantees the
    env-level half.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    cmd = ["uv", "run", rel_entrypoint, "-c", str(toml_path), *steps]
    print(f"[tier-a] cwd={repo.name}  cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=repo, env=env, check=True)


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
                f"shape mismatch: port=(dim={p_dim}, raw_dim={p_raw}), "
                f"upstream=(dim={u_dim}, raw_dim={u_raw})"
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
            result["detail"] = (
                f"{mf} missing: port={(port_dir / mf).exists()}, "
                f"upstream={(upstream_dir / mf).exists()}"
            )
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
            f"(ensure dump_rotator=true in both TOMLs)"
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
            "matching PCA and medoid. Check rotator_seed plumbing on "
            "both sides (should be 42)."
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
            extra = (
                f" (demoted: {art.get('pca_demotion_reason')}; "
                f"mean_Δ={mean_d:.3e}, comp_Δ={comp_d:.3e})"
            )
        elif status == "FAIL" and art.get("detail"):
            extra = f" ({art['detail']})"
        print(f"  [{status:>12}] {stage:>8}  {file_}{extra}")
    print("===================================\n")


# ── Entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port-config", type=Path, required=True,
                   help="AlayaLite alignment-mode TOML (examples/laser/configs/*_alayaP.toml)")
    p.add_argument("--upstream-config", type=Path, required=True,
                   help="Laser alignment-mode TOML (Laser/reproduce/configs/*_origP.toml)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Root dir for per-side outputs + JSON diff report")
    p.add_argument("--report", type=Path, default=None,
                   help="JSON report path (default: <out-root>/tier_a_report.json)")
    p.add_argument("--skip-run", action="store_true",
                   help="Compare existing <out-root>/{port,upstream} artifacts; skip pipeline invocation.")
    p.add_argument("--port-repo", type=Path, default=ALAYA_REPO_DEFAULT,
                   help=f"AlayaLite repo root (default: {ALAYA_REPO_DEFAULT})")
    p.add_argument("--upstream-repo", type=Path, default=LASER_REPO_DEFAULT,
                   help=f"Upstream Laser repo root (default: {LASER_REPO_DEFAULT})")
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
                f"[tier-a][err] dataset.{field} mismatch: "
                f"port={port_v!r}, upstream={up_v!r}",
                file=sys.stderr,
            )
            return EXIT_HARNESS_ERR
    name = port_cfg["dataset"]["name"]
    degree = int(port_cfg["dataset"]["degree"])
    main_dim = int(port_cfg["dataset"]["main_dimension"])

    # Rewrite TOMLs for per-side output.
    port_toml = args.out_root / f"{name}_port.toml"
    upstream_toml = args.out_root / f"{name}_upstream.toml"
    rewrite_output(args.port_config, port_toml, str(port_out))
    rewrite_output(args.upstream_config, upstream_toml, str(upstream_out))

    if not args.skip_run:
        try:
            # Port: vamana (shared graph) + pca + medoid + index.
            run_pipeline(
                args.port_repo, "examples/laser/main.py", port_toml,
                ["vamana", "pca", "medoid", "index"],
            )
            # Upstream: pca + medoid + index (no vamana step upstream).
            run_pipeline(
                args.upstream_repo, "reproduce/main.py", upstream_toml,
                ["pca", "medoid", "index"],
            )
        except subprocess.CalledProcessError as e:
            print(f"[tier-a][err] pipeline failed (exit={e.returncode}): {e.cmd}", file=sys.stderr)
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
