"""
Laser CLI wrapper around ``alayalite.laser.Index.fit``.

Two steps:
- ``build``  — runs the full PCA/medoid/Vamana/QG pipeline via ``Index.fit``.
- ``search`` — loads the built index and runs an EF sweep.

Usage:
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml all
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search --threads 4 --efs 100 200 300
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml -c examples/laser/configs/sift.toml all
"""

import argparse
import builtins
import functools
import gc
import os
import sys
from math import log10
from time import time

# tomllib is stdlib from Python 3.11 onwards; fall back to the tomli backport
# on 3.9 / 3.10 (the oldest supported interpreters per pyproject.toml).
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import numpy as np  # pylint: disable=wrong-import-position
import psutil  # pylint: disable=wrong-import-position

# Keep print and C++ stdout in sync.
builtins.print = functools.partial(builtins.print, flush=True)

# pylint: disable=invalid-name
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

if not sys.stdout.isatty():
    DIM = BOLD = GREEN = CYAN = YELLOW = RED = RESET = ""


def header(text, width=60):
    print(f"\n{BOLD}{CYAN}" + "=" * width + f"{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}" + "=" * width + f"{RESET}")


def step_header(name, width=40):
    print(f"\n{BOLD}>> {name.upper()}{RESET}")
    print(f"{DIM}" + "─" * width + f"{RESET}")


def info(tag, msg):
    print(f"  {DIM}[{tag}]{RESET} {msg}")


def success(tag, msg):
    print(f"  {GREEN}[{tag}]{RESET} {msg}")


def warn(tag, msg):
    print(f"  {YELLOW}[{tag}]{RESET} {msg}")


def separator(width=56):
    print(f"  {DIM}" + "─" * width + f"{RESET}")


def beam_size_gen(k):
    assert k >= 1
    e = max(0, int(log10(k)) - 1)
    index = 0
    bases = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    while True:
        yield bases[index] * int(10**e)
        index += 1
        if index == len(bases):
            e += 1
            index = 0


def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024


STEPS = ["build", "search"]

# DEFAULTS for all optional fields. The `build_vamana_*` entries mirror
# `alaya::vamana::kDefaultVamanaBuildParams` in
# `include/index/graph/vamana/build_dispatch.hpp` — keep in lockstep.
DEFAULTS = {
    "topk": 10,
    "threads": 1,
    "beam_width": 16,
    "dram_budget": 1.0,
    "ep_num": 300,
    "efs": [80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500],
    "build_threads": 48,
    "ef_indexing": 200,
    "warmup": 10,
    "runs": 30,
    "build_vamana_L": 200,
    "build_vamana_alpha": 1.2,
    "build_vamana_seed": 1234,
    "build_vamana_num_threads": 0,
    "build_vamana_dram_budget_gb": 32.0,
}


def print_config(cfg):
    # pylint: disable=inconsistent-quotes
    info(
        "config",
        f"metric={cfg['metric']}  degree={cfg['degree']}  main_dim={cfg['main_dimension']}",
    )
    info(
        "build",
        f"threads={cfg['build_threads']}  ef_indexing={cfg['ef_indexing']}",
    )
    info(
        "search",
        f"topk={cfg['topk']}  threads={cfg['threads']}  bw={cfg['beam_width']}  "
        f"dram={cfg['dram_budget']}GB  warmup={cfg['warmup']}  runs={cfg['runs']}",
    )


# ── Args & Config ──


def parse_args():
    parser = argparse.ArgumentParser(
        description="Laser reproduce pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        action="append",
        help="Path to dataset config TOML file (can be specified multiple times)",
    )
    parser.add_argument(
        "steps",
        nargs="+",
        choices=STEPS + ["all"],
        help="Steps to run: vamana, pca, medoid, index, search, or all",
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--dram-budget", type=float, default=None)
    parser.add_argument("--efs", nargs="+", type=int, default=None)
    parser.add_argument("--ep-num", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--main-dim", type=int, default=None)
    parser.add_argument("--build-threads", type=int, default=None)
    parser.add_argument("--ef-indexing", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)

    return parser.parse_args()


def load_config(toml_path, cli_args):
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    ds = raw["dataset"]
    paths = raw["paths"]
    build = raw.get("build", {})
    search = raw.get("search", {})
    build_vamana = raw.get("build_vamana", {})

    # Three-way R contract: Vamana build R, TOML [dataset].degree, and
    # Laser degree_bound must match. R is sourced from [dataset].degree;
    # permitting R under [build_vamana] would let it diverge from the
    # other two sites. See proposal D7.
    if "R" in build_vamana:
        raise ValueError(
            "R must be set via [dataset].degree; remove R from [build_vamana] — "
            "see proposal integrate-vamana-into-laser-pipeline D7"
        )

    def resolve_search(key, cli_val):
        if cli_val is not None:
            return cli_val
        return search.get(key, DEFAULTS[key])

    def resolve_build(key, cli_val):
        if cli_val is not None:
            return cli_val
        # TOML root takes precedence over [build] so seed/thread invariants
        # (pca_seed, build_threads, ...) stay visible at a glance.
        if key in raw:
            return raw[key]
        return build.get(key, DEFAULTS[key])

    def resolve_build_vamana(toml_key, defaults_key):
        return build_vamana.get(toml_key, DEFAULTS[defaults_key])

    pca_seed = raw.get("pca_seed")
    medoid_seed = raw.get("medoid_seed")
    rotator_seed = int(raw.get("rotator_seed", 0))
    force_single_thread = bool(raw.get("force_single_thread", False))
    dump_rotator = bool(raw.get("dump_rotator", False))

    build_threads_resolved = resolve_build("build_threads", cli_args.build_threads)

    # `qg_builder`'s `#pragma omp parallel for num_threads(num_threads_)`
    # overrides OMP_NUM_THREADS, so process-level single-thread pinning is
    # silently defeated unless build_threads is also 1. Fail loudly.
    if force_single_thread and int(build_threads_resolved) != 1:
        raise ValueError(
            f"force_single_thread=true requires build_threads=1 (got build_threads={build_threads_resolved!r})."
        )

    # force_single_thread signals "I want byte-reproducible builds". Any
    # unseeded RNG silently defeats that intent, so require all three
    # seeds explicitly rather than falling back to defaults.
    if force_single_thread:
        missing_seeds = [
            key
            for key, val in (
                ("pca_seed", pca_seed),
                ("medoid_seed", medoid_seed),
            )
            if val is None
        ]
        if rotator_seed == 0:
            missing_seeds.append("rotator_seed (nonzero)")
        if missing_seeds:
            raise ValueError(f"force_single_thread=true requires explicit seeds, missing: {missing_seeds}")

    # dump_rotator with an unseeded rotator produces a different file
    # every run, which is useless for the byte-equality comparison that
    # dump_rotator exists to support. Catch this at config load.
    if dump_rotator and rotator_seed == 0:
        raise ValueError(
            "dump_rotator=true requires rotator_seed ≠ 0 (unseeded rotator produces a different dump every run)."
        )

    name = ds["name"]
    output = paths["output"]

    # [paths].vamana is optional: when omitted, step_vamana writes to a
    # derived path under {output}/data/{name}/vamana/graph.index. Existing
    # configs that pin a path (e.g. gist_diskann.toml's externally-built
    # graph) continue to work unchanged — step_vamana's idempotence check
    # detects the valid file and skips the build.
    vamana_path = paths.get("vamana")
    if vamana_path is None:
        vamana_path = f"{output}/data/{name}/vamana/graph.index"

    return {
        "name": name,
        "metric": ds["metric"],
        "degree": cli_args.degree or ds["degree"],
        "main_dimension": cli_args.main_dim or ds["main_dimension"],
        "base": paths["base"],
        "query": paths["query"],
        "gt": paths["gt"],
        "vamana": vamana_path,
        "output": output,
        "build_threads": build_threads_resolved,
        "ef_indexing": resolve_build("ef_indexing", cli_args.ef_indexing),
        "topk": resolve_search("topk", cli_args.topk),
        "threads": resolve_search("threads", cli_args.threads),
        "beam_width": resolve_search("beam_width", cli_args.beam_width),
        "dram_budget": resolve_search("dram_budget", cli_args.dram_budget),
        "ep_num": resolve_search("ep_num", cli_args.ep_num),
        "efs": resolve_search("efs", cli_args.efs),
        "warmup": resolve_search("warmup", cli_args.warmup),
        "runs": resolve_search("runs", cli_args.runs),
        "build_vamana_L": resolve_build_vamana("L", "build_vamana_L"),
        "build_vamana_alpha": resolve_build_vamana("alpha", "build_vamana_alpha"),
        "build_vamana_seed": resolve_build_vamana("seed", "build_vamana_seed"),
        "build_vamana_num_threads": resolve_build_vamana("num_threads", "build_vamana_num_threads"),
        "build_vamana_dram_budget_gb": resolve_build_vamana("dram_budget_gb", "build_vamana_dram_budget_gb"),
        "pca_seed": pca_seed,
        "medoid_seed": medoid_seed,
        "rotator_seed": rotator_seed,
        "force_single_thread": force_single_thread,
        "dump_rotator": dump_rotator,
        "seed": raw.get("seed"),
    }


def data_dir(cfg):
    return f"{cfg['output']}/data/{cfg['name']}"  # pylint: disable=inconsistent-quotes


# ── Steps ──


def _fit_unified(cfg, *, auto_load: bool = False):
    # pylint: disable=import-outside-toplevel
    from alayalite import laser

    ds_name = cfg["name"]
    return laser.Index.fit(
        cfg["base"],
        output_dir=data_dir(cfg),
        name=f"dsqg_{ds_name}",
        build_params=laser.BuildParams(
            metric=cfg["metric"],
            main_dim=cfg["main_dimension"],
            R=cfg["degree"],
            L=cfg["build_vamana_L"],
            alpha=cfg["build_vamana_alpha"],
            ef_indexing=cfg["ef_indexing"],
            ep_num=cfg["ep_num"],
        ),
        num_threads=cfg["build_threads"],
        seed=42 if cfg["seed"] is None else int(cfg["seed"]),
        dram_budget_gb=cfg["build_vamana_dram_budget_gb"],
        skip_existing=True,
        auto_load=auto_load,
    )


# DiskANN single-file .index header: 24 bytes total.
#   offset 0..7   uint64  expected_file_size
#   offset 8..11  uint32  max_observed_degree  (= R)
#   offset 12..15 uint32  start                (medoid id)
#   offset 16..23 uint64  frozen_pts           (0 in this port)
_VAMANA_HEADER_BYTES = 24


def _read_vamana_header(path):
    """Return (file_size, max_degree, start, frozen_pts) or None if unreadable."""
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return None
    if file_size < _VAMANA_HEADER_BYTES:
        return None
    with open(path, "rb") as f:
        buf = f.read(_VAMANA_HEADER_BYTES)
    if len(buf) != _VAMANA_HEADER_BYTES:
        return None
    expected_size = int.from_bytes(buf[0:8], "little")
    max_degree = int.from_bytes(buf[8:12], "little")
    start = int.from_bytes(buf[12:16], "little")
    frozen_pts = int.from_bytes(buf[16:24], "little")
    return file_size, expected_size, max_degree, start, frozen_pts


def step_build(cfg):
    name = cfg["name"]
    info(name, "Building/validating LASER artifacts via Index.fit(...)")
    t1 = time()
    _fit_unified(cfg, auto_load=False)
    success(name, f"Build done in {time() - t1:.1f}s")


def _find_efs(index, query, gt, nq, topk):
    efs = []
    gen = beam_size_gen(topk)
    prev_recall = 0

    while True:
        ef = next(gen)
        efs.append(ef)
        index.set_params(ef_search=ef, beam_width=16)

        total_time = 0
        results = []
        for i in range(nq):
            t1 = time()
            pred = index.search(query[i], topk)
            t2 = time()
            results.append(pred)
            total_time += t2 - t1

        total_correct = sum(1 for i in range(nq) for j in range(topk) if gt[i][j] in set(results[i]))
        recall = total_correct / (nq * topk) * 100
        qps = nq / total_time

        if recall > 99.8 or (recall - prev_recall) < 0.05 or qps < 10:
            break
        prev_recall = recall

    return efs


def step_search(cfg):
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from alayalite import laser
    from alayalite.laser._io import read_fbin, read_ibin

    name = cfg["name"]
    ddir = data_dir(cfg)
    degree = cfg["degree"]
    md = cfg["main_dimension"]
    topk = cfg["topk"]
    num_threads = cfg["threads"]
    bw = cfg["beam_width"]
    num_warmup = cfg["warmup"]
    num_runs = cfg["runs"]
    single_search = num_threads == 1

    info(name, "Loading data...")
    query = read_fbin(cfg["query"])
    gt = read_ibin(cfg["gt"])

    NQ = query.shape[0]  # pylint: disable=invalid-name
    info(name, f"Queries: {NQ:,}, GT: {gt.shape}")

    info(name, f"Loading index R{degree}_MD{md} from prefix...")
    m1 = get_memory_usage()
    index = laser.Index.from_prefix(
        f"{ddir}/dsqg_{name}",
        dram_budget_gb=cfg["dram_budget"],
    )
    memory = get_memory_usage() - m1
    info(name, f"Index loaded, memory: {memory:.1f} MB")

    cur_efs = cfg["efs"] if len(cfg["efs"]) > 0 else _find_efs(index, query, gt, NQ, topk)
    info(name, f"EFS: {cur_efs}")
    info(name, f"Warmup: {num_warmup} rounds, Runs: {num_runs} per EF")
    info(name, "Running benchmark...\n")

    # Print table header
    print(f"  {BOLD}{'EF':>6}  {'QPS':>10}  {'Recall':>8}  {'Latency(us)':>12}  {'P99.9(us)':>10}{RESET}")  # pylint: disable=inconsistent-quotes
    separator()

    all_qps, all_recall, all_lat, all_p99 = [], [], [], []

    for ef in cur_efs:
        total_time = 0
        latencies = []

        index.set_params(ef_search=ef, num_threads=num_threads, beam_width=bw)

        # Warmup: pre-heat page cache, SSD controller, and OS I/O path
        for _ in range(num_warmup):
            index.batch_search(query, topk)

        for _ in range(num_runs):
            if single_search:
                results = []
                for i in range(NQ):
                    t1 = time()
                    pred = index.search(query[i], topk)
                    t2 = time()
                    results.append(pred)
                    total_time += t2 - t1
                    latencies.append((t2 - t1) * 1e6)
            else:
                t1 = time()
                results = index.batch_search(query, topk)
                t2 = time()
                total_time += t2 - t1
                latencies.append(0)

        total_correct = sum(1 for i in range(NQ) for j in range(topk) if gt[i][j] in set(results[i]))

        qps = NQ * num_runs / total_time
        recall = total_correct / (NQ * topk) * 100
        mean_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99.9)

        # Print row immediately
        recall_color = GREEN if recall >= 95 else YELLOW if recall >= 90 else RED
        print(
            f"  {ef:>6d}  {qps:>10.1f}  {recall_color}{recall:>7.2f}%{RESET}  {mean_lat:>12.1f}  {p99_lat:>10.1f}",
            flush=True,
        )

        all_qps.append(qps)
        all_recall.append(recall)
        all_lat.append(mean_lat)
        all_p99.append(p99_lat)

    separator()
    print(f"  Memory: {memory:.1f} MB")

    df = pd.DataFrame(
        {
            "QPS": all_qps,
            "Recall": all_recall,
            "EFS": cur_efs,
            "Mean Latency (us)": all_lat,
            "P99.9 Latency (us)": all_p99,
            "Method": f"dsqg_R{degree}_MD{md}",
            "Memory": memory,
        }
    )

    res_dir = f"{cfg['output']}/results/{name}/dsqg/"  # pylint: disable=inconsistent-quotes
    os.makedirs(res_dir, exist_ok=True)
    csv_path = f"{res_dir}dsqg_R{degree}_MD{md}_TOP{topk}_T{num_threads}.csv"
    df.to_csv(csv_path, index=False)
    success(name, f"Saved to {csv_path}")

    del index, df
    gc.collect()


STEP_FUNCS = {
    "build": step_build,
    "search": step_search,
}


_SINGLE_THREAD_LIMITER = None


def apply_single_thread_invariant():
    """Force the process into single-thread mode and assert observed state.

    Byte-reproducible builds need OpenMP, BLAS/MKL, and faiss all pinned
    at 1 thread BEFORE any parallel region runs (IncrementalPCA, faiss
    k-means, QG link). Env-level OMP_NUM_THREADS=1 covers most cases,
    but faiss needs a direct `omp_set_num_threads(1)` call, and
    `qg_builder`'s `num_threads(num_threads_)` pragma is only neutralised
    by `build_threads=1` in the TOML (see config-load fail-fast above).

    Asserts the observed thread counts all land at 1; raises on mismatch.
    """
    import faiss  # pylint: disable=import-outside-toplevel
    import threadpoolctl  # pylint: disable=import-outside-toplevel

    faiss.omp_set_num_threads(1)
    # Keep a module-level reference to the limiter so it outlives this
    # function. `threadpool_limits` is a context manager; the limit is
    # removed via `__exit__`, and some threadpoolctl versions wire that
    # into `__del__`. Holding the reference sidesteps both and keeps the
    # single-thread pin in place for every downstream pipeline step.
    global _SINGLE_THREAD_LIMITER  # noqa: PLW0603
    _SINGLE_THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=1)

    faiss_n = faiss.omp_get_max_threads()
    omp_n = None
    blas_report = []
    for pool in threadpoolctl.threadpool_info():
        if pool["user_api"] == "openmp":
            omp_n = pool["num_threads"]
        if pool["user_api"] == "blas":
            blas_report.append((pool["prefix"], pool["num_threads"]))

    print(
        f"[single-thread self-check] faiss.omp_get_max_threads={faiss_n}  "
        f"openmp.num_threads={omp_n}  "
        f"blas={blas_report}",
        flush=True,
    )

    assert faiss_n == 1, f"faiss.omp_get_max_threads() = {faiss_n}, expected 1"
    assert omp_n == 1, f"openmp.num_threads = {omp_n}, expected 1"
    for prefix, n in blas_report:
        assert n == 1, f"blas/{prefix} num_threads = {n}, expected 1"


def main():
    args = parse_args()
    steps = STEPS if "all" in args.steps else args.steps

    for toml_path in args.config:
        cfg = load_config(toml_path, args)
        header(f"{cfg['name']}  ({toml_path})")  # pylint: disable=inconsistent-quotes
        print_config(cfg)

        if cfg["force_single_thread"]:
            apply_single_thread_invariant()

        for step in steps:
            step_header(step)
            STEP_FUNCS[step](cfg)

    print()


if __name__ == "__main__":
    main()
