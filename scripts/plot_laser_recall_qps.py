"""Plot recall-QPS Pareto curves for the Laser port validation runs.

Reads the baseline + validation search CSVs under
``build_graph/laser_port/{baseline,validation}_<date>/<combo>/results/<name>/
dsqg/dsqg_R64_MD256_TOP10_T1.csv`` and emits a PNG showing QPS vs recall@10
for all (dataset × vamana) × (baseline / validation) combinations.

Usage:
    uv run scripts/plot_laser_recall_qps.py \\
        --baseline-dir /md1/huangliang/alaya-dev/build_graph/laser_port/baseline_20260421 \\
        --port-out-dir /md1/huangliang/alaya-dev/build_graph/laser_port/validation_20260421 \\
        --out /md1/huangliang/alaya-dev/build_graph/laser_port/validation_20260421/recall_qps.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMBOS: list[tuple[str, str, str]] = [
    # (combo_dir_key, toml_dataset_name, display label)
    ("synth_alayaV",  "synth_100k_512d", "synth + AlayaV"),
    ("synth_diskV",   "synth_100k_512d", "synth + DiskV"),
    ("gist1m_alayaV", "gist",            "gist1m + AlayaV"),
    ("gist1m_diskV",  "gist",            "gist1m + DiskV"),
    ("bigcode_alayaV", "bigcode",        "bigcode + AlayaV"),
    ("bigcode_diskV",  "bigcode",        "bigcode + DiskV"),
    ("cohere_alayaV",  "cohere",         "cohere + AlayaV"),
    ("cohere_diskV",   "cohere",         "cohere + DiskV"),
]

CSV_RELATIVE = "results/{name}/dsqg/dsqg_R64_MD256_TOP10_T1.csv"


def _load(combo_dir: Path, name: str) -> pd.DataFrame | None:
    p = combo_dir / CSV_RELATIVE.format(name=name)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.sort_values("EFS").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--port-out-dir", type=Path, required=True)
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
            ba = _load(args.baseline_dir / combo_key, name)
            po = _load(args.port_out_dir / combo_key, name)
            color = "tab:blue" if "AlayaV" in label else "tab:orange"
            if ba is not None:
                ax.plot(ba["Recall"], ba["QPS"], color=color, linestyle="-",
                        marker="o", markersize=5,
                        label=f"{label} (baseline)")
            if po is not None:
                ax.plot(po["Recall"], po["QPS"], color=color, linestyle="--",
                        marker="x", markersize=6,
                        label=f"{label} (ported)")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Laser port — recall/QPS sweep (EF ∈ {80,90,100,110,130,150,200,250,300,400,500})",
        y=0.995,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
