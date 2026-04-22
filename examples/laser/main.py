"""
Laser Reproduce Pipeline

Usage:
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml all
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml search --threads 4 --efs 100 200 300
    uv run examples/laser/main.py -c examples/laser/configs/gist.toml -c examples/laser/configs/sift.toml all
"""

import argparse
import gc
import os
import sys
from time import time

# tomllib is stdlib from Python 3.11 onwards; fall back to the tomli backport
# on older interpreters (AlayaLite supports 3.8+ per pyproject.toml).
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import numpy as np

from alayalite.laser.pretty import (
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    header,
    info,
    separator,
    step_header,
    success,
    warn,
)

STEPS = ["pca", "medoid", "index", "search"]

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
}


def print_config(cfg):
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
        f"topk={cfg['topk']}  threads={cfg['threads']}  bw={cfg['beam_width']}  dram={cfg['dram_budget']}GB  warmup={cfg['warmup']}  runs={cfg['runs']}",
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
        help="Steps to run: pca, medoid, index, search, or all",
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

    def resolve_search(key, cli_val):
        if cli_val is not None:
            return cli_val
        return search.get(key, DEFAULTS[key])

    def resolve_build(key, cli_val):
        if cli_val is not None:
            return cli_val
        return build.get(key, DEFAULTS[key])

    return {
        "name": ds["name"],
        "metric": ds["metric"],
        "degree": cli_args.degree or ds["degree"],
        "main_dimension": cli_args.main_dim or ds["main_dimension"],
        "base": paths["base"],
        "query": paths["query"],
        "gt": paths["gt"],
        "vamana": paths["vamana"],
        "output": paths["output"],
        "build_threads": resolve_build("build_threads", cli_args.build_threads),
        "ef_indexing": resolve_build("ef_indexing", cli_args.ef_indexing),
        "topk": resolve_search("topk", cli_args.topk),
        "threads": resolve_search("threads", cli_args.threads),
        "beam_width": resolve_search("beam_width", cli_args.beam_width),
        "dram_budget": resolve_search("dram_budget", cli_args.dram_budget),
        "ep_num": resolve_search("ep_num", cli_args.ep_num),
        "efs": resolve_search("efs", cli_args.efs),
        "warmup": resolve_search("warmup", cli_args.warmup),
        "runs": resolve_search("runs", cli_args.runs),
    }


def data_dir(cfg):
    return f"{cfg['output']}/data/{cfg['name']}"


# ── Steps ──


def step_pca(cfg):
    from alayalite.laser.io import read_fbin
    from alayalite.laser.pca import (
        fit_incremental_pca,
        load_pca_params,
        pca_transform_and_save,
        sample_vectors_from_fbin,
        save_pca_params,
    )

    name = cfg["name"]
    ddir = data_dir(cfg)
    pca_base_path = f"{ddir}/dsqg_{name}_pca_base.fbin"
    pca_params_path = f"{ddir}/dsqg_{name}_pca.bin"

    # Check if PCA transform is already complete
    if os.path.exists(pca_base_path):
        file_size = os.path.getsize(pca_base_path)
        if file_size >= 8:
            with open(pca_base_path, "rb") as f:
                header = np.fromfile(f, dtype=np.int32, count=2)
            expected_size = 8 + int(header[0]) * int(header[1]) * 4
            if file_size >= expected_size:
                warn(name, "PCA files already exist. Skipping.")
                return
        # Incomplete file - resume if PCA params are saved
        if os.path.exists(pca_params_path):
            info(name, f"Incomplete PCA base detected ({file_size:,} bytes). Resuming...")
            vectors = read_fbin(cfg["base"])
            pca = load_pca_params(pca_params_path)
            t1 = time()
            pca_transform_and_save(vectors, pca, pca_base_path)
            success(name, f"PCA resume done in {time() - t1:.1f}s -> {pca_base_path}")
            return

    os.makedirs(ddir, exist_ok=True)
    info(name, f"Loading base vectors from {cfg['base']}")
    # Seed the PCA training-sample selection so baselines are byte-reproducible.
    # Upstream Laser leaves np.random.choice on the uninitialised global state,
    # which produces different PCA components on every run; see
    # openspec/changes/port-laser-disk-index/design.md D10.
    vectors, sample_vecs = sample_vectors_from_fbin(cfg["base"], seed=cfg.get("pca_seed", 42))
    _, d = vectors.shape
    info(
        name, f"Vectors: {vectors.shape[0]:,} x {d}d, samples: {sample_vecs.shape[0]:,}"
    )

    t1 = time()
    pca = fit_incremental_pca(sample_vecs, n_components=d)
    save_pca_params(pca, pca_params_path)
    pca_transform_and_save(vectors, pca, pca_base_path)
    success(name, f"PCA done in {time() - t1:.1f}s -> {pca_base_path}")


def step_medoid(cfg):
    from alayalite.laser.medoid import generate_and_save_medoids

    name = cfg["name"]
    ddir = data_dir(cfg)
    base_path = f"{ddir}/dsqg_{name}_pca_base.fbin"

    if not os.path.exists(base_path):
        warn(name, "PCA base not found. Run 'pca' step first.")
        return

    indices_path = f"{ddir}/dsqg_{name}_medoids_indices"
    vectors_path = f"{ddir}/dsqg_{name}_medoids"

    if os.path.exists(indices_path) and os.path.exists(vectors_path):
        warn(name, "Medoids already exist. Skipping.")
        return

    info(name, f"Generating {cfg['ep_num']} medoids...")
    t1 = time()
    generate_and_save_medoids(
        base_path, indices_path, vectors_path, cfg["ep_num"],
        seed=cfg.get("pca_seed", 42),
    )
    success(name, f"Medoid done in {time() - t1:.1f}s")


def step_index(cfg):
    from alayalite import laser
    from alayalite.laser.io import read_fbin

    name = cfg["name"]
    ddir = data_dir(cfg)
    degree = cfg["degree"]
    md = cfg["main_dimension"]

    build_threads = cfg["build_threads"]
    ef_indexing = cfg["ef_indexing"]

    info(
        name,
        f"Building index R{degree}_MD{md} (threads={build_threads}, ef={ef_indexing})...",
    )
    base = read_fbin(f"{ddir}/dsqg_{name}_pca_base.fbin", use_mmap=True)
    N, D = base.shape
    info(name, f"Vectors: {N:,} x {D}d")

    t1 = time()
    index = laser.Index(
        index_type="QG",
        metric=cfg["metric"],
        num_elements=N,
        main_dimension=md,
        dimension=D,
        degree_bound=degree,
    )
    index_path = f"{ddir}/dsqg_{name}"
    index.build_index(
        cfg["vamana"], index_path, EF=ef_indexing, num_thread=build_threads
    )
    success(name, f"Index R{degree}_MD{md} done in {time() - t1:.1f}s")


def _find_efs(index, query, gt, nq, topk):
    from alayalite.laser.beam_size import beam_size_gen

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

        total_correct = sum(
            1 for i in range(nq) for j in range(topk) if gt[i][j] in set(results[i])
        )
        recall = total_correct / (nq * topk) * 100
        qps = nq / total_time

        if recall > 99.8 or (recall - prev_recall) < 0.05 or qps < 10:
            break
        prev_recall = recall

    return efs


def step_search(cfg):
    import numpy as np
    import pandas as pd
    from alayalite import laser
    from alayalite.laser.io import read_fbin, read_ibin
    from alayalite.laser.memory import get_memory_usage

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
    base = read_fbin(f"{ddir}/dsqg_{name}_pca_base.fbin")
    query = read_fbin(cfg["query"])
    gt = read_ibin(cfg["gt"])

    NQ = query.shape[0]
    N, D = base.shape
    info(name, f"Base: {N:,} x {D}d, Queries: {NQ:,}, GT: {gt.shape}")

    info(name, f"Loading index R{degree}_MD{md}...")
    m1 = get_memory_usage()
    index = laser.Index(
        index_type="QG",
        metric=cfg["metric"],
        num_elements=N,
        main_dimension=md,
        dimension=D,
        degree_bound=degree,
    )
    index_path = f"{ddir}/dsqg_{name}"
    index.load(index_path, cfg["dram_budget"])
    memory = get_memory_usage() - m1
    info(name, f"Index loaded, memory: {memory:.1f} MB")

    cur_efs = (
        cfg["efs"] if len(cfg["efs"]) > 0 else _find_efs(index, query, gt, NQ, topk)
    )
    info(name, f"EFS: {cur_efs}")
    info(name, f"Warmup: {num_warmup} rounds, Runs: {num_runs} per EF")
    info(name, "Running benchmark...\n")

    # Print table header
    print(
        f"  {BOLD}{'EF':>6}  {'QPS':>10}  {'Recall':>8}  "
        f"{'Latency(us)':>12}  {'P99.9(us)':>10}{RESET}"
    )
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

        total_correct = sum(
            1 for i in range(NQ) for j in range(topk) if gt[i][j] in set(results[i])
        )

        qps = NQ * num_runs / total_time
        recall = total_correct / (NQ * topk) * 100
        mean_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99.9)

        # Print row immediately
        recall_color = GREEN if recall >= 95 else YELLOW if recall >= 90 else RED
        print(
            f"  {ef:>6d}  "
            f"{qps:>10.1f}  "
            f"{recall_color}{recall:>7.2f}%{RESET}  "
            f"{mean_lat:>12.1f}  "
            f"{p99_lat:>10.1f}",
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

    res_dir = f"{cfg['output']}/results/{name}/dsqg/"
    os.makedirs(res_dir, exist_ok=True)
    csv_path = f"{res_dir}dsqg_R{degree}_MD{md}_TOP{topk}_T{num_threads}.csv"
    df.to_csv(csv_path, index=False)
    success(name, f"Saved to {csv_path}")

    del index, df
    gc.collect()


STEP_FUNCS = {
    "pca": step_pca,
    "medoid": step_medoid,
    "index": step_index,
    "search": step_search,
}


def main():
    args = parse_args()
    steps = STEPS if "all" in args.steps else args.steps

    for toml_path in args.config:
        cfg = load_config(toml_path, args)
        header(f"{cfg['name']}  ({toml_path})")
        print_config(cfg)

        for step in steps:
            step_header(step)
            STEP_FUNCS[step](cfg)

    print()


if __name__ == "__main__":
    main()
