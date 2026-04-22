"""Plot recall-QPS overlay comparing unpinned validation vs NUMA-node-0
pinned runs for the Laser port.

Produces a 4-panel figure (one per dataset) where each curve pair shows
the same (combo) measured without pinning (solid) and with
`numactl --cpunodebind=0 --membind=0` (dashed). The right gap between
solid and dashed at the same recall is the NUMA-locality speed-up.

Usage:
    uv run scripts/plot_numactl_comparison.py \\
        --unpinned  .../validation_20260421 \\
        --pinned    .../numactl_node0_20260421 \\
        --out       .../numactl_node0_20260421/recall_qps_numactl.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMBOS: list[tuple[str, str, str]] = [
    ("synth_alayaV",   "synth_100k_512d", "synth + AlayaV"),
    ("synth_diskV",    "synth_100k_512d", "synth + DiskV"),
    ("gist1m_alayaV",  "gist",            "gist1m + AlayaV"),
    ("gist1m_diskV",   "gist",            "gist1m + DiskV"),
    ("bigcode_alayaV", "bigcode",         "bigcode + AlayaV"),
    ("bigcode_diskV",  "bigcode",         "bigcode + DiskV"),
    ("cohere_alayaV",  "cohere",          "cohere + AlayaV"),
    ("cohere_diskV",   "cohere",          "cohere + DiskV"),
]

CSV_RELATIVE = "results/{name}/dsqg/dsqg_R64_MD256_TOP10_T1.csv"


def _load(root: Path, combo_key: str, name: str) -> pd.DataFrame | None:
    p = root / combo_key / CSV_RELATIVE.format(name=name)
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("EFS").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpinned", type=Path, required=True)
    ap.add_argument("--pinned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    datasets = [
        ("synth_100k_512d", "synth_100k_512d (100K × 512d)"),
        ("gist",             "gist1m (1M × 960d)"),
        ("bigcode",          "bigcode (10.4M × 768d)"),
        ("cohere",           "cohere (10.1M × 768d)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=False)
    for ax, (ds_key, title) in zip(axes.flat, datasets, strict=True):
        ax.set_title(f"Recall@10 vs QPS — {title}")
        ax.set_xlabel("Recall@10 (%)")
        ax.set_ylabel("QPS (queries/s, threads=1)")
        ax.grid(True, alpha=0.3)

        for combo_key, name, label in COMBOS:
            if name != ds_key:
                continue
            un = _load(args.unpinned, combo_key, name)
            pn = _load(args.pinned,   combo_key, name)
            color = "tab:blue" if "AlayaV" in label else "tab:orange"
            if un is not None:
                ax.plot(un["Recall"], un["QPS"], color=color, linestyle="-",
                        marker="o", markersize=5, alpha=0.85,
                        label=f"{label} (unpinned)")
            if pn is not None:
                ax.plot(pn["Recall"], pn["QPS"], color=color, linestyle="--",
                        marker="x", markersize=6, alpha=0.9,
                        label=f"{label} (numactl node0)")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Laser port — unpinned vs NUMA-node-0 pinned search (EF sweep {80…500})",
        y=0.995,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
