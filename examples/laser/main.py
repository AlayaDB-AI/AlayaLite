# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Small CLI wrapper around ``alayalite.laser.Index.fit``.

Two steps:
- ``build``  — runs the unified PCA/medoid/Vamana/QG build.
- ``search`` — loads the built index and runs an EF sweep.

Usage:
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml all
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search --threads 4 --efs 100 200 300
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
        description="Laser unified Index.fit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        action="append",
        help="Path to dataset config TOML file (can be specified multiple times)",
    )
    parser.add_argument("steps", nargs="+", choices=STEPS + ["all"], help="Steps to run: build, search, or all")
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

    legacy_top_level = {"pca_seed", "medoid_seed", "rotator_seed", "force_single_thread", "dump_rotator"} & raw.keys()
    if legacy_top_level:
        legacy_family = "pca_seed, medoid_seed, rotator_seed, force_single_thread, dump_rotator"
        found_list = ", ".join(sorted(legacy_top_level))
        raise ValueError(
            "legacy per-step alignment fields are no longer supported by examples/laser/main.py "
            f"(found: {found_list}; retired family: {legacy_family}); "
            "use the unified top-level seed field instead"
        )

    if "vamana" in paths:
        raise ValueError(
            "legacy [paths].vamana is no longer supported by examples/laser/main.py; "
            "the unified Index.fit wrapper builds and validates its own Vamana artifact"
        )

    retired_build_vamana = {"seed", "num_threads"} & build_vamana.keys()
    if retired_build_vamana:
        found_list = ", ".join(sorted(retired_build_vamana))
        raise ValueError(
            "legacy [build_vamana] seed/num_threads fields are no longer supported by examples/laser/main.py "
            f"(found: {found_list}); use top-level seed and [build].build_threads"
        )

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
        # TOML root takes precedence over [build] so thread settings stay
        # visible at a glance in compact configs.
        if key in raw:
            return raw[key]
        return build.get(key, DEFAULTS[key])

    def resolve_build_vamana(toml_key, defaults_key):
        return build_vamana.get(toml_key, DEFAULTS[defaults_key])

    build_threads_resolved = resolve_build("build_threads", cli_args.build_threads)

    name = ds["name"]
    output = paths["output"]

    return {
        "name": name,
        "metric": ds["metric"],
        "degree": cli_args.degree or ds["degree"],
        "main_dimension": cli_args.main_dim or ds["main_dimension"],
        "base": paths["base"],
        "query": paths["query"],
        "gt": paths["gt"],
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
        "build_vamana_dram_budget_gb": resolve_build_vamana("dram_budget_gb", "build_vamana_dram_budget_gb"),
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


def _warmup_index(index, query, *, topk, rounds, single_search):
    if single_search:
        for _ in range(rounds):
            for i in range(query.shape[0]):
                index.search(query[i], topk)
    else:
        for _ in range(rounds):
            index.batch_search(query, topk)


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

        # Warmup must exercise the same API path that the timed loop uses.
        _warmup_index(index, query, topk=topk, rounds=num_warmup, single_search=single_search)

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


def main():
    args = parse_args()
    steps = STEPS if "all" in args.steps else args.steps

    for toml_path in args.config:
        cfg = load_config(toml_path, args)
        header(f"{cfg['name']}  ({toml_path})")  # pylint: disable=inconsistent-quotes
        print_config(cfg)

        for step in steps:
            step_header(step)
            STEP_FUNCS[step](cfg)

    print()


if __name__ == "__main__":
    main()
