#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""
GIST-1M: Recall@10 vs QPS -- 6-curve comparison, Vamana L=100 vs L=200

Data embedded for reproducibility.
Run: uv run python experiments/laser_cross_bench/plot_recall_qps_l200.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# embedded data
# ---------------------------------------------------------------------------

# Each row: (ef, qps, recall_at_10)
DATA: dict[str, list[tuple[int, float, float]]] = {
    "orig_diskann": [
        (80, 964.91, 0.9165),
        (90, 934.07, 0.9267),
        (100, 898.75, 0.9348),
        (110, 867.03, 0.9420),
        (130, 816.99, 0.9526),
        (150, 770.60, 0.9595),
        (200, 682.96, 0.9731),
        (250, 614.03, 0.9798),
        (300, 558.18, 0.9842),
        (400, 475.14, 0.9897),
        (500, 413.15, 0.9927),
    ],
    "orig_lite": [
        # ef=80 is a cold-cache outlier -- dropped, start from ef=90
        (90, 954.81, 0.9002),
        (100, 913.52, 0.9111),
        (110, 883.83, 0.9239),
        (130, 830.88, 0.9369),
        (150, 791.99, 0.9476),
        (200, 704.06, 0.9628),
        (250, 634.14, 0.9726),
        (300, 575.93, 0.9775),
        (400, 492.00, 0.9836),
        (500, 426.20, 0.9876),
    ],
    "orig_lite_l200": [
        # ef=80 is a cold-cache outlier -- dropped, start from ef=90
        (90, 943.84, 0.9251),
        (100, 910.43, 0.9334),
        (110, 879.79, 0.9394),
        (130, 840.91, 0.9521),
        (150, 780.90, 0.9591),
        (200, 692.07, 0.9723),
        (250, 616.36, 0.9800),
        (300, 557.61, 0.9846),
        (400, 483.03, 0.9901),
        (500, 420.16, 0.9930),
    ],
    "lite_diskann": [
        (80, 948.86, 0.9074),
        (90, 922.60, 0.9178),
        (100, 886.89, 0.9283),
        (110, 873.26, 0.9364),
        (130, 814.51, 0.9495),
        (150, 763.03, 0.9566),
        (200, 674.21, 0.9679),
        (250, 605.22, 0.9772),
        (300, 552.68, 0.9813),
        (400, 475.06, 0.9877),
        (500, 413.32, 0.9906),
    ],
    "lite_lite": [
        # ef=80 is a cold-cache outlier -- dropped, start from ef=90
        (90, 940.34, 0.8975),
        (100, 904.08, 0.9093),
        (110, 880.90, 0.9175),
        (130, 841.47, 0.9334),
        (150, 789.62, 0.9459),
        (200, 708.63, 0.9596),
        (250, 634.36, 0.9698),
        (300, 572.11, 0.9746),
        (400, 497.94, 0.9820),
        (500, 431.60, 0.9859),
    ],
    "lite_lite_l200": [
        # ef=80 is a cold-cache outlier -- dropped, start from ef=90
        (90, 923.46, 0.9193),
        (100, 884.23, 0.9292),
        (110, 839.19, 0.9364),
        (130, 794.00, 0.9490),
        (150, 739.10, 0.9581),
        (200, 662.44, 0.9698),
        (250, 591.39, 0.9771),
        (300, 537.43, 0.9829),
        (400, 453.76, 0.9882),
        (500, 410.50, 0.9911),
    ],
}

SERIES_STYLE: dict[str, dict] = {
    "orig_diskann": {
        "label": "Orig + DiskANN (L=200)",
        "color": "#2196F3",
        "linestyle": "-",
        "marker": "o",
    },
    "orig_lite": {
        "label": "Orig + AlayaLite (L=100)",
        "color": "#90CAF9",
        "linestyle": "--",
        "marker": "s",
    },
    "orig_lite_l200": {
        "label": "Orig + AlayaLite (L=200)",
        "color": "#0D47A1",
        "linestyle": "-.",
        "marker": "^",
    },
    "lite_diskann": {
        "label": "AlayaLite + DiskANN (L=200)",
        "color": "#FF9800",
        "linestyle": "-",
        "marker": "o",
    },
    "lite_lite": {
        "label": "AlayaLite + AlayaLite (L=100)",
        "color": "#FFCC80",
        "linestyle": "--",
        "marker": "s",
    },
    "lite_lite_l200": {
        "label": "AlayaLite + AlayaLite (L=200)",
        "color": "#E65100",
        "linestyle": "-.",
        "marker": "^",
    },
}

OUTPUT_PATH = Path("/md1/huangliang/alaya-dev/tmp/laser_cross_bench/results/recall_qps_l200.png")

# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for key, rows in DATA.items():
        style = SERIES_STYLE[key]
        recalls = [r[2] for r in rows]
        qps = [r[1] for r in rows]
        efs = [r[0] for r in rows]

        ax.plot(
            recalls,
            qps,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.8,
            label=style["label"],
        )

        # Annotate only ef=500 (curve tail label)
        for ef, recall, q in zip(efs, recalls, qps):
            if ef == 500:
                ax.annotate(
                    f"ef={ef}",
                    xy=(recall, q),
                    xytext=(-42, 8),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=style["color"],
                )

    ax.set_title(
        "GIST-1M: Recall@10 vs QPS — Vamana L=100 vs L=200",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (queries/sec)", fontsize=11)
    ax.grid(color="#CCCCCC", linewidth=0.6, linestyle="-", alpha=0.7)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
