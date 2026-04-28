"""Plots for the `align-diskann-sharded-with-upstream` Tier A + Tier B results.

Outputs PNGs into
`/md1/huangliang/alaya-dev/data/build_graph/diskann_sharded_alignment/figures/`.

All numbers are taken from the captured evidence under
`data/build_graph/diskann_sharded_alignment/{tier_a_20260424,tier_a_gist1m_20260425}/`.
Run via:

    cd /md1/huangliang/alaya-dev/AlayaLite
    uv run python experiments/diskann_sharded_alignment_plots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("/md1/huangliang/alaya-dev/data/build_graph/diskann_sharded_alignment/figures")


@dataclass(frozen=True)
class TierARun:
    label: str
    wall_s: float
    num_parts: int
    medoid_sha_short: str


SYNTH_ALAYA = TierARun("AlayaLite (seed=1234)", 185, 13, "7546708c…")
SYNTH_DISKANN = TierARun("Patched DiskANN (matched kwargs)", 234, 13, "7546708c…")
GIST_ALAYA = TierARun("AlayaLite (seed=1234)", 2669, 43, "ec6840b4…")
GIST_DISKANN = TierARun("Patched DiskANN (matched kwargs)", 2666, 43, "ec6840b4…")

GIST_UNPATCHED = [
    ("run 1", 3085, 41),
    ("run 2", 3041, 43),
    ("run 3", 2649, 49),
]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 120,
        }
    )


def plot_tier_b_envelope():
    """Tier B: 3 unpatched DiskANN GIST-1M runs vs AlayaLite seeded."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    runs = [r[0] for r in GIST_UNPATCHED] + ["AlayaLite\n(seeded)"]
    parts = [r[2] for r in GIST_UNPATCHED] + [GIST_ALAYA.num_parts]
    colors = ["#888888"] * 3 + ["#1f6fa3"]

    lo, hi = min(parts[:3]), max(parts[:3])
    ax.axhspan(lo, hi, color="#cce5ff", alpha=0.35, label=f"Tier B envelope [{lo}, {hi}]")

    bars = ax.bar(runs, parts, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, parts):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.6, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("num_parts (growth-loop converged value)")
    ax.set_title("Tier B envelope — GIST-1M, R=64, build_dram_budget=0.5 GiB, T=1")
    ax.set_ylim(0, max(parts) + 6)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color="#888888", label="Unpatched DiskANN (random_device)"),
        mpatches.Patch(color="#1f6fa3", label="AlayaLite (seed=1234)"),
        mpatches.Patch(color="#cce5ff", alpha=0.6, label=f"Envelope [{lo}, {hi}]"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    ax.text(
        0.02,
        0.97,
        f"AlayaLite num_parts={GIST_ALAYA.num_parts} ∈ [{lo}, {hi}] — Tier B passes",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#eaffea", "edgecolor": "#7ba37b"},
    )

    fig.tight_layout()
    out = FIG_DIR / "fig1_tier_b_envelope_gist1m.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_tier_a_artifact_matrix():
    """Tier A artifact-class match matrix on both datasets."""
    artifacts = ["_medoids.bin", "_centroids.bin", "merged .index", "per-shard\n_subshard-*_mem.index"]
    datasets = ["synth_100k_512d\n(num_parts=13)", "GIST-1M\n(num_parts=43)"]

    # 2 = byte-equal, 1 = structural/header equal, 0 = divergent (expected)
    matrix = np.array(
        [
            [2, 2],
            [1, 1],
            [1, 1],
            [0, 0],
        ]
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    cmap = plt.matplotlib.colors.ListedColormap(["#fbd6d6", "#fff4cc", "#cdebcd"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    labels = {0: "byte ✗\n(per-shard scope)", 1: "structural ✓\nbytes ⚠", 2: "byte-equal ✓"}
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(
                j, i, labels[int(v)], ha="center", va="center", fontsize=10, fontweight="bold" if v == 2 else "normal"
            )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(artifacts)))
    ax.set_yticklabels(artifacts)
    ax.set_title("Tier A artifact-class comparison — AlayaLite vs Patched DiskANN")

    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(artifacts), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    legend_patches = [
        mpatches.Patch(color="#cdebcd", label="byte-equal (mechanical invariant proven)"),
        mpatches.Patch(color="#fff4cc", label="structural / header equal, bytes diverge"),
        mpatches.Patch(color="#fbd6d6", label="per-shard byte divergence — deferred"),
    ]
    ax.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)

    fig.tight_layout()
    out = FIG_DIR / "fig2_tier_a_artifact_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_wall_time_comparison():
    """Wall-time per build, both datasets, both pipelines, plus 3 unpatched."""
    fig, (ax_synth, ax_gist) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})

    # ------- left: synth_100k_512d -------
    syn_labels = ["AlayaLite\n(seeded)", "Patched DiskANN\n(aligned)"]
    syn_walls = [SYNTH_ALAYA.wall_s, SYNTH_DISKANN.wall_s]
    bars = ax_synth.bar(
        syn_labels, syn_walls, color=["#1f6fa3", "#cb6f1c"], edgecolor="black", linewidth=0.5, width=0.55
    )
    for b, v in zip(bars, syn_walls):
        ax_synth.text(
            b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f} s", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax_synth.set_ylabel("Wall time (seconds, T=1)")
    ax_synth.set_title("synth_100k_512d (num_parts=13)")
    ax_synth.set_ylim(0, max(syn_walls) * 1.18)
    ax_synth.grid(axis="y", linestyle=":", alpha=0.5)

    # ------- right: GIST-1M (alaya + aligned + 3 unpatched) -------
    gist_labels = [
        "AlayaLite\n(seeded)",
        "Patched DiskANN\n(aligned)",
        *[f"Unpatched\n{r[0]}" for r in GIST_UNPATCHED],
    ]
    gist_walls = [GIST_ALAYA.wall_s, GIST_DISKANN.wall_s, *[r[1] for r in GIST_UNPATCHED]]
    gist_colors = ["#1f6fa3", "#cb6f1c", "#888888", "#888888", "#888888"]
    bars = ax_gist.bar(gist_labels, gist_walls, color=gist_colors, edgecolor="black", linewidth=0.5, width=0.55)
    for b, v in zip(bars, gist_walls):
        ax_gist.text(
            b.get_x() + b.get_width() / 2,
            v + 50,
            f"{v:.0f} s\n({v / 60:.1f} min)",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax_gist.set_ylabel("Wall time (seconds, T=1)")
    ax_gist.set_title("GIST-1M (extended-marker scenario)")
    ax_gist.set_ylim(0, max(gist_walls) * 1.18)
    ax_gist.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Wall-time per build — single-thread (T=1) on each dataset", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig3_wall_time_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_patch_sites_overview():
    """Visual summary of patched RNG/logic sites and their alignment role."""
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")

    box_w = 36
    pad = 2
    col_x = [4, 4 + box_w + pad, 4 + 2 * (box_w + pad)]

    sites = [
        (
            "gen_random_slice\n(file→mem · inputdata→mem)",
            "uint64_t seed = 0",
            "Sample RNG aligned;\nat seed != 0 uses mt19937_64 +\nuniform<double>, mirroring AlayaLite",
            (col_x[0], 70),
            "#1f6fa3",
        ),
        (
            "partition_with_ram_budget",
            "uint64_t seed = 0",
            "Threads seed into both gen_random_slice\ncalls and kmeans++; rng-by-ref overload\ngives AlayaLite continuous stream",
            (col_x[1], 70),
            "#1f6fa3",
        ),
        (
            "kmeanspp_selecting_pivots\n+ selecting_pivots (fallback)",
            "uint64_t seed = 0",
            "Both internal random_device sites\nseeded; kmeans++ / Lloyd's now\ndeterministic given pivot input",
            (col_x[2], 70),
            "#1f6fa3",
        ),
        (
            "disk_utils.cpp::merge_shards",
            "shuffle_seed · drop_self_loops\nforced_global_medoid",
            "shuffle_seed ⇒ mt19937 std::shuffle\ndrop_self_loops ⇒ AlayaLite semantics\nforced_global_medoid ⇒ header override",
            (col_x[0], 30),
            "#cb6f1c",
        ),
        (
            "build_merged_vamana_index",
            "all of the above\n+ keep_intermediates",
            "Top-level entry threads kwargs;\nkeep_intermediates leaves per-shard\n_mem.index for SHA comparison",
            (col_x[1], 30),
            "#cb6f1c",
        ),
        (
            "apps/build_merged_vamana_standalone.cpp",
            "(NEW BINARY) all 5 alignment\nflags exposed via Boost.po",
            "PQ-skipping CLI; runs the same\npipeline as build_disk_index\nwithout the ~11h PQ training step",
            (col_x[2], 30),
            "#5a9b59",
        ),
    ]

    for title, kwarg, body, (x, y), color in sites:
        rect = mpatches.FancyBboxPatch(
            (x, y - 22),
            box_w,
            26,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + 1.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)
        ax.text(
            x + box_w / 2,
            y - 8,
            kwarg,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#333333",
            style="italic",
            family="monospace",
        )
        ax.text(x + box_w / 2, y - 16.5, body, ha="center", va="center", fontsize=8.5, color="#555555")

    ax.text(
        60,
        96,
        "Patched DiskANN sites for AlayaLite alignment (branch align-diskann-sharded-with-alaya)",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.text(
        60,
        91,
        "All kwargs default-preserving (seed=0 → upstream-main bytes); non-zero/non-default → AlayaLite semantics",
        ha="center",
        fontsize=10,
        color="#444444",
        style="italic",
    )

    legend = [
        mpatches.Patch(facecolor="white", edgecolor="#1f6fa3", label="Partition-stage RNG"),
        mpatches.Patch(facecolor="white", edgecolor="#cb6f1c", label="Merge-stage logic / RNG"),
        mpatches.Patch(facecolor="white", edgecolor="#5a9b59", label="CLI surface"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

    out = FIG_DIR / "fig4_patch_sites_overview.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_recall_and_qps():
    """Tier B Recall-vs-QPS Pareto curves on GIST-1M (canonical ANN-benchmark form).

    Each variant is one curve; points along the curve correspond to
    L_search ∈ {10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500}.
    Higher recall is to the right; higher QPS is up. AlayaLite (seeded)
    and Patched DiskANN (aligned) curves overlap → search behavior is
    Pareto-equivalent. Unpatched DiskANN runs (random_device path) are
    drawn as background context; their natural variance defines the
    Tier B envelope around the aligned curve.

    Data captured 2026-04-25 with `search_memory_index --T 4 -K 10`
    against `gist_query.fbin` + `gist_gt.ibin`. Full per-L tables in
    `/tmp/search_results/gist/<tag>.full.log`.
    """
    L_values = [10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500]

    # 11-point sweep, T=4 search threads, K=10.
    runs = [
        (
            "AlayaLite (seeded)",
            [60.20, 74.43, 81.69, 89.01, 92.31, 95.15, 97.25, 98.23, 99.09, 99.41, 99.64],
            [8054.43, 5444.30, 4159.63, 2875.13, 2143.79, 1656.91, 1172.24, 909.02, 570.51, 469.16, 359.24],
            "#1f6fa3",
            "o",
            2.4,
            8,
        ),
        (
            "Patched DiskANN (aligned)",
            [61.10, 74.06, 81.46, 89.28, 92.49, 95.14, 97.14, 98.11, 99.08, 99.39, 99.57],
            [7924.58, 5474.37, 4170.04, 2860.54, 1898.59, 1441.03, 1043.22, 828.00, 606.02, 476.53, 408.00],
            "#cb6f1c",
            "s",
            2.4,
            8,
        ),
        (
            "Unpatched run 1",
            [60.64, 74.41, 81.55, 88.95, 92.27, 94.87, 97.29, 98.25, 99.06, 99.37, 99.54],
            [7637.10, 4870.14, 3635.73, 2507.22, 1714.27, 1302.62, 1056.03, 828.92, 537.78, 418.63, 384.87],
            "#a8a8a8",
            "^",
            1.0,
            5,
        ),
        (
            "Unpatched run 2",
            [60.74, 74.64, 81.82, 88.72, 92.10, 95.06, 97.14, 98.15, 99.05, 99.34, 99.54],
            [5566.06, 5144.48, 4080.92, 2580.26, 1881.41, 1444.90, 1048.91, 836.57, 605.43, 495.34, 412.67],
            "#888888",
            "v",
            1.0,
            5,
        ),
        (
            "Unpatched run 3",
            [61.57, 74.44, 81.36, 88.66, 92.20, 94.84, 96.98, 97.95, 98.88, 99.23, 99.39],
            [7306.13, 5283.46, 4072.32, 2793.24, 1985.37, 1418.91, 1003.62, 809.43, 593.29, 528.80, 418.17],
            "#666666",
            "D",
            1.0,
            5,
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 6.5))

    for name, recall, qps, color, marker, lw, ms in runs:
        ax.plot(
            recall,
            qps,
            color=color,
            linewidth=lw,
            marker=marker,
            markersize=ms,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=name,
            alpha=0.95,
        )
        # Annotate L_search at a few canonical points on the AlayaLite curve
        # so the reader can read off "what L produces this recall".
        if name == "AlayaLite (seeded)":
            for L, r, q in zip(L_values, recall, qps):
                # Annotate every other point to avoid clutter
                if L in (10, 30, 70, 100, 200, 500):
                    ax.annotate(
                        f"L={L}",
                        (r, q),
                        textcoords="offset points",
                        xytext=(8, 8),
                        fontsize=8.5,
                        color=color,
                        fontweight="bold",
                    )

    ax.set_xlabel("Recall@10 (%)  →  better")
    ax.set_ylabel("Queries per second (T=4)  →  better")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.set_title("Recall vs QPS — GIST-1M, K=10, L_search ∈ {10..500} (11-point sweep)")
    # Curves go upper-left → lower-right. Legend in upper-right.
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    # Tier B verdict — max |Δrecall| over the full sweep, plus per-L
    # summary at the canonical recall regimes (low/mid/high).
    alaya_recall = runs[0][1]
    aligned_recall = runs[1][1]
    deltas = [abs(a - b) for a, b in zip(alaya_recall, aligned_recall)]
    max_delta = max(deltas)
    max_idx = deltas.index(max_delta)
    text = (
        f"Tier B verdict — AlayaLite vs Patched DiskANN (aligned)\n"
        f"  max |Δrecall| over 11-point sweep : {max_delta:.2f} pp  (at L={L_values[max_idx]})\n"
        f"  |Δrecall| @ L=10   : {deltas[0]:.2f} pp\n"
        f"  |Δrecall| @ L=100  : {deltas[5]:.2f} pp\n"
        f"  |Δrecall| @ L=500  : {deltas[-1]:.2f} pp\n"
        f"  Threshold: ≤ 1.0 pp  →  satisfied at every L"
    )
    ax.text(
        0.02,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8.8,
        family="monospace",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eaffea", "edgecolor": "#7ba37b"},
    )

    fig.tight_layout()
    out = FIG_DIR / "fig5_recall_qps_gist1m.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_laser_downstream():
    """Laser-side Recall vs QPS — sharded Vamana fed into the port-Laser pipeline.

    AlayaV / DiskV refer to the *Vamana producer*; both pipelines feed
    their merged sharded graph into the same port-Laser
    (PCA → medoid → QG index → DSQG search). If the partition-stage
    byte-equality (`_medoids.bin` SHA match) plus structural parity at
    the merged-graph layer is enough, the two Pareto curves should
    overlap on the Laser side.

    Captured 2026-04-25 from
    `examples/laser/configs/gist1m_alignshard_{alayaV,diskV}_alayaP.toml`
    runs at T=1 search, K=10, ef ∈ {30..500} (12-point sweep).
    """
    ef_values = [30, 40, 50, 60, 80, 100, 130, 170, 220, 300, 400, 500]
    runs = [
        (
            "AlayaLite Vamana → port-Laser (alayaV)",
            [72.66, 79.65, 84.25, 87.24, 90.91, 92.89, 94.91, 96.34, 97.07, 97.95, 98.55, 98.86],
            [1213.5, 1105.1, 880.8, 1058.1, 955.6, 884.2, 795.4, 726.8, 649.2, 559.5, 481.0, 417.8],
            "#1f6fa3",
            "o",
            2.4,
            8,
        ),
        (
            "Patched DiskANN aligned Vamana → port-Laser (diskV)",
            [73.01, 79.55, 83.74, 86.80, 90.56, 92.80, 94.73, 96.22, 97.02, 97.80, 98.42, 98.72],
            [1250.3, 1158.8, 1073.8, 1010.6, 991.5, 920.5, 824.0, 752.7, 670.1, 573.8, 491.0, 429.8],
            "#cb6f1c",
            "s",
            2.4,
            8,
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    for name, recall, qps, color, marker, lw, ms in runs:
        ax.plot(
            recall,
            qps,
            color=color,
            linewidth=lw,
            marker=marker,
            markersize=ms,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=name,
            alpha=0.95,
        )
    # Annotate ef values on the alayaV curve at canonical points
    for ef, r, q in zip(ef_values, runs[0][1], runs[0][2]):
        if ef in (30, 60, 100, 170, 300, 500):
            ax.annotate(
                f"ef={ef}",
                (r, q),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8.5,
                color=runs[0][3],
                fontweight="bold",
            )

    ax.set_xlabel("Recall@10 (%)  →  better")
    ax.set_ylabel("Queries per second (T=1, beam=16)  →  better")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.set_title("Laser downstream Recall vs QPS — GIST-1M, port-Laser pipeline")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    deltas = [abs(a - b) for a, b in zip(runs[0][1], runs[1][1])]
    max_delta = max(deltas)
    max_idx = deltas.index(max_delta)
    text = (
        f"Laser downstream verdict — alayaV vs diskV (port-Laser fixed)\n"
        f"  max |Δrecall| over 12-point sweep : {max_delta:.2f} pp  (at ef={ef_values[max_idx]})\n"
        f"  |Δrecall| @ ef=30   : {deltas[0]:.2f} pp\n"
        f"  |Δrecall| @ ef=100  : {deltas[5]:.2f} pp\n"
        f"  |Δrecall| @ ef=500  : {deltas[-1]:.2f} pp\n"
        f"  Threshold: ≤ 1.0 pp  →  satisfied at every ef"
    )
    ax.text(
        0.02,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8.8,
        family="monospace",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eaffea", "edgecolor": "#7ba37b"},
    )

    fig.tight_layout()
    out = FIG_DIR / "fig6_laser_downstream_gist1m.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_tier_b_envelope()
    plot_tier_a_artifact_matrix()
    plot_wall_time_comparison()
    plot_patch_sites_overview()
    plot_recall_and_qps()
    plot_laser_downstream()


if __name__ == "__main__":
    main()
