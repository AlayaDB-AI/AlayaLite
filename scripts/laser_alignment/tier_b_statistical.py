# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tier B statistical-equivalence harness for the Laser-upstream alignment gate.

Compares the 4-combo GIST 1 M seed-injected experiment:
    A = alayaV × alayaP
    B = alayaV × origP
    C = diskV  × alayaP
    D = diskV  × origP

For every EF in the sweep and every one of the 6 pair combinations
({A,B,C,D} choose 2), the gate requires:
    * |ΔRecall|        ≤ 0.3 pp
    * |ΔQPS| / min(QPS) ≤ 5 %

Additionally, a Pareto-dominance check at each EF: no combo's
(recall, QPS) point may dominate another's simultaneously by more
than (+0.1 pp, +2 %).

Usage:
    uv run python scripts/laser_alignment/tier_b_statistical.py \\
        --csv-a  <alayaV_alayaP>/results/gist/dsqg/dsqg_R64_MD256_TOP10_T1.csv \\
        --csv-b  <alayaV_origP>/results/gist/dsqg/dsqg_R64_MD256_TOP10_T1.csv \\
        --csv-c  <diskV_alayaP>/results/gist/dsqg/dsqg_R64_MD256_TOP10_T1.csv \\
        --csv-d  <diskV_origP>/results/gist/dsqg/dsqg_R64_MD256_TOP10_T1.csv \\
        --out-dir <build_graph/laser_alignment/tier_b_YYYYMMDD>

Exit codes:
    0  PASS
    1  threshold exceeded (recall or QPS)
    2  Pareto dominance violated
    3  harness error (missing CSV, malformed, EF set mismatch)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECALL_THRESHOLD_PP: float = 0.3
QPS_REL_THRESHOLD: float = 0.05
PARETO_RECALL_EPS_PP: float = 0.1
PARETO_QPS_EPS_REL: float = 0.02

# Canonical EF sweep for the Tier B 4-combo GIST 1 M experiment.
# All 4 CSVs must cover exactly this set; override via --expected-efs
# for datasets beyond GIST 1 M.
CANONICAL_EFS: tuple[int, ...] = (30, 40, 50, 60, 80, 100, 130, 170, 220, 300, 400, 500)

EXIT_PASS = 0
EXIT_THRESHOLD_FAIL = 1
EXIT_PARETO_FAIL = 2
EXIT_HARNESS_ERR = 3

COMBO_LABELS = ["A", "B", "C", "D"]
COMBO_NAMES = {
    "A": "alayaV × alayaP",
    "B": "alayaV × origP",
    "C": "diskV  × alayaP",
    "D": "diskV  × origP",
}


# ── CSV ingestion ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComboRun:
    label: str
    csv_path: Path
    # EF → (QPS, Recall)
    points: dict[int, tuple[float, float]]


def load_combo(label: str, csv_path: Path) -> ComboRun:
    if not csv_path.exists():
        raise FileNotFoundError(f"[tier-b] combo {label} CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"QPS", "Recall", "EFS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[tier-b] combo {label} CSV at {csv_path} missing columns: {missing}")
    pts: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        ef = int(row["EFS"])
        qps = float(row["QPS"])
        rec = float(row["Recall"])
        if ef in pts:
            raise ValueError(
                f"[tier-b] combo {label} CSV at {csv_path} has duplicate EF={ef}; refusing to silently overwrite"
            )
        if qps <= 0 or rec < 0:
            raise ValueError(
                f"[tier-b] combo {label} CSV at {csv_path} row EF={ef}: non-positive metric QPS={qps}, Recall={rec}"
            )
        pts[ef] = (qps, rec)
    return ComboRun(label=label, csv_path=csv_path, points=pts)


# ── Gate checks ───────────────────────────────────────────────────────────


@dataclass
class PairViolation:
    ef: int
    pair: tuple[str, str]
    metric: str  # "recall_pp", "qps_rel", or "pareto"
    value: float
    threshold: float
    detail: str


def check_thresholds(combos: dict[str, ComboRun], efs: list[int]) -> list[PairViolation]:
    """Per-EF, per-pair |ΔR| ≤ 0.3 pp and |ΔQPS|/min(QPS) ≤ 5 %."""
    violations: list[PairViolation] = []
    for ef in efs:
        for ci, cj in itertools.combinations(COMBO_LABELS, 2):
            qi, ri = combos[ci].points[ef]
            qj, rj = combos[cj].points[ef]
            d_rec = abs(ri - rj)
            if d_rec > RECALL_THRESHOLD_PP:
                violations.append(
                    PairViolation(
                        ef=ef,
                        pair=(ci, cj),
                        metric="recall_pp",
                        value=d_rec,
                        threshold=RECALL_THRESHOLD_PP,
                        detail=f"|Recall_{ci} - Recall_{cj}| = {d_rec:.3f} pp ({ri:.3f} vs {rj:.3f})",
                    )
                )
            q_min = min(qi, qj)
            d_qps_rel = abs(qi - qj) / q_min if q_min > 0 else float("inf")
            if d_qps_rel > QPS_REL_THRESHOLD:
                violations.append(
                    PairViolation(
                        ef=ef,
                        pair=(ci, cj),
                        metric="qps_rel",
                        value=d_qps_rel,
                        threshold=QPS_REL_THRESHOLD,
                        detail=f"|QPS_{ci} - QPS_{cj}| / min = {d_qps_rel * 100:.2f} % ({qi:.1f} vs {qj:.1f})",
                    )
                )
    return violations


def check_pareto_dominance(combos: dict[str, ComboRun], efs: list[int]) -> list[PairViolation]:
    """For each EF, no combo SHALL dominate another by >(+0.1 pp, +2 %).

    Dominance here means strictly higher recall AND strictly higher QPS,
    both exceeding the ε-ball. Non-positive QPS inputs are guarded
    against to avoid division-by-zero; such inputs should already have
    been rejected in `load_combo()`.
    """
    violations: list[PairViolation] = []
    for ef in efs:
        for ci, cj in itertools.combinations(COMBO_LABELS, 2):
            qi, ri = combos[ci].points[ef]
            qj, rj = combos[cj].points[ef]
            if qi <= 0 or qj <= 0:
                continue  # guarded in load_combo; belt-and-braces
            # i dominates j?
            if (ri - rj > PARETO_RECALL_EPS_PP) and (qi - qj) / qj > PARETO_QPS_EPS_REL:
                violations.append(
                    PairViolation(
                        ef=ef,
                        pair=(ci, cj),
                        metric="pareto",
                        value=max(ri - rj, (qi - qj) / qj),
                        threshold=max(PARETO_RECALL_EPS_PP, PARETO_QPS_EPS_REL),
                        detail=f"{ci} dominates {cj} at EF={ef}: "
                        f"ΔR={ri - rj:+.3f}pp, ΔQPS={((qi - qj) / qj) * 100:+.2f}%",
                    )
                )
            elif (rj - ri > PARETO_RECALL_EPS_PP) and (qj - qi) / qi > PARETO_QPS_EPS_REL:
                violations.append(
                    PairViolation(
                        ef=ef,
                        pair=(ci, cj),
                        metric="pareto",
                        value=max(rj - ri, (qj - qi) / qi),
                        threshold=max(PARETO_RECALL_EPS_PP, PARETO_QPS_EPS_REL),
                        detail=f"{cj} dominates {ci} at EF={ef}: "
                        f"ΔR={rj - ri:+.3f}pp, ΔQPS={((qj - qi) / qi) * 100:+.2f}%",
                    )
                )
    return violations


# ── Output renderers (PNG + markdown) ────────────────────────────────────


def render_markdown(
    combos: dict[str, ComboRun],
    efs: list[int],
    thr_violations: list[PairViolation],
    pareto_violations: list[PairViolation],
    out_path: Path,
) -> None:
    """Mirror data/gist1m_seed_test/SUMMARY.md table format."""
    lines: list[str] = []
    lines.append("# Tier B Statistical Gate — 4-combo Summary")
    lines.append("")
    lines.append(
        f"Thresholds: |ΔRecall| ≤ {RECALL_THRESHOLD_PP} pp, "
        f"|ΔQPS|/min ≤ {QPS_REL_THRESHOLD * 100:.0f} %, "
        f"Pareto ε = (±{PARETO_RECALL_EPS_PP} pp, ±{PARETO_QPS_EPS_REL * 100:.0f} %)."
    )
    lines.append("")
    lines.append("## Combos")
    lines.append("")
    for lbl in COMBO_LABELS:
        lines.append(f"- **{lbl}**: {COMBO_NAMES[lbl]} — `{combos[lbl].csv_path}`")
    lines.append("")
    lines.append("## Recall / QPS per EF (mirror of data/gist1m_seed_test/SUMMARY.md)")
    lines.append("")
    # One cell per combo, format "Recall % / QPS" — matches the reference
    # SUMMARY.md Pareto table layout.
    header = "| EFS | " + " | ".join(f"{lbl} {COMBO_NAMES[lbl]}" for lbl in COMBO_LABELS) + " |"
    sep = "|-----|" + "|".join(["-" * 20] * len(COMBO_LABELS)) + "|"
    lines.append(header)
    lines.append(sep)
    for ef in efs:
        row_cells = [f"{ef:>4d}"]
        for lbl in COMBO_LABELS:
            qps, rec = combos[lbl].points[ef]
            row_cells.append(f"{rec:>6.2f} % / {qps:>6.0f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if not thr_violations and not pareto_violations:
        lines.append("**PASS** — all 6 pairs within thresholds at every EF; no Pareto dominance.")
    else:
        lines.append("**FAIL**:")
        if thr_violations:
            lines.append("")
            lines.append(f"### Threshold violations ({len(thr_violations)})")
            lines.append("")
            lines.append("| EF | pair | metric | value | detail |")
            lines.append("|----|------|--------|-------|--------|")
            for v in thr_violations:
                lines.append(f"| {v.ef} | {v.pair[0]}–{v.pair[1]} | {v.metric} | {v.value:.4g} | {v.detail} |")
        if pareto_violations:
            lines.append("")
            lines.append(f"### Pareto violations ({len(pareto_violations)})")
            lines.append("")
            lines.append("| EF | pair | detail |")
            lines.append("|----|------|--------|")
            for v in pareto_violations:
                lines.append(f"| {v.ef} | {v.pair[0]}–{v.pair[1]} | {v.detail} |")
    out_path.write_text("\n".join(lines) + "\n")


def render_plot(combos: dict[str, ComboRun], efs: list[int], out_path: Path) -> None:
    """Per-EF sweep plot: QPS vs Recall for all 4 combos."""
    # Import matplotlib lazily — the laser extras group pins it but keep
    # the harness import light when only generating markdown.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    markers = {"A": "o", "B": "s", "C": "^", "D": "D"}
    for lbl in COMBO_LABELS:
        qps_arr = [combos[lbl].points[ef][0] for ef in efs]
        rec_arr = [combos[lbl].points[ef][1] for ef in efs]
        ax.plot(rec_arr, qps_arr, marker=markers[lbl], label=f"{lbl}: {COMBO_NAMES[lbl]}")
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("QPS")
    # Linear scale — GIST 1 M QPS range is typically ~3× (400-1200); log-scale
    # would compress the ±5 % threshold differences and make envelope drift
    # visually harder to read. Use --log-qps to re-enable when the range
    # widens (e.g., 10 M datasets).
    ax.set_title("Tier B — 4-combo Recall/QPS envelope")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-a", type=Path, required=True, help="alayaV × alayaP run CSV")
    p.add_argument("--csv-b", type=Path, required=True, help="alayaV × origP run CSV")
    p.add_argument("--csv-c", type=Path, required=True, help="diskV × alayaP run CSV")
    p.add_argument("--csv-d", type=Path, required=True, help="diskV × origP run CSV")
    p.add_argument("--out-dir", type=Path, required=True, help="Report dir (writes report.md, summary.json, curve.png)")
    p.add_argument(
        "--expected-efs",
        type=int,
        nargs="+",
        default=list(CANONICAL_EFS),
        help=f"Fixed EF sweep to validate against (default: {list(CANONICAL_EFS)})",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        combos: dict[str, ComboRun] = {
            "A": load_combo("A", args.csv_a),
            "B": load_combo("B", args.csv_b),
            "C": load_combo("C", args.csv_c),
            "D": load_combo("D", args.csv_d),
        }
    except (FileNotFoundError, ValueError) as e:
        print(f"[tier-b][err] {e}", file=sys.stderr)
        return EXIT_HARNESS_ERR

    # Require identical EF sweep across all 4 combos AND match to the
    # canonical sweep pinned in spec.md. Tier B has no tolerance for
    # schema drift between runs NOR for a silently-shortened sweep that
    # misses the recall regime where drift is most visible.
    expected = set(args.expected_efs)
    ef_sets = {lbl: set(combos[lbl].points.keys()) for lbl in COMBO_LABELS}
    for lbl in COMBO_LABELS:
        missing = expected - ef_sets[lbl]
        extra = ef_sets[lbl] - expected
        if missing or extra:
            print(
                f"[tier-b][err] combo {lbl} EF sweep mismatch vs expected: "
                f"missing={sorted(missing)}, extra={sorted(extra)}",
                file=sys.stderr,
            )
            return EXIT_HARNESS_ERR
    efs = sorted(expected)

    thr_violations = check_thresholds(combos, efs)
    pareto_violations = check_pareto_dominance(combos, efs)

    # Write outputs.
    report_md = args.out_dir / "report.md"
    summary_json = args.out_dir / "summary.json"
    curve_png = args.out_dir / "curve.png"
    render_markdown(combos, efs, thr_violations, pareto_violations, report_md)
    try:
        render_plot(combos, efs, curve_png)
    except ImportError as e:
        print(f"[tier-b][warn] skipped plot (matplotlib missing): {e}", file=sys.stderr)
    summary_json.write_text(
        json.dumps(
            {
                "efs": efs,
                "combos": {lbl: str(c.csv_path) for lbl, c in combos.items()},
                "thresholds": {
                    "recall_pp": RECALL_THRESHOLD_PP,
                    "qps_rel": QPS_REL_THRESHOLD,
                    "pareto_recall_pp": PARETO_RECALL_EPS_PP,
                    "pareto_qps_rel": PARETO_QPS_EPS_REL,
                },
                "violations": {
                    "threshold": [
                        {
                            "ef": v.ef,
                            "pair": list(v.pair),
                            "metric": v.metric,
                            "value": v.value,
                            "threshold": v.threshold,
                            "detail": v.detail,
                        }
                        for v in thr_violations
                    ],
                    "pareto": [
                        {
                            "ef": v.ef,
                            "pair": list(v.pair),
                            "metric": v.metric,
                            "value": v.value,
                            "threshold": v.threshold,
                            "detail": v.detail,
                        }
                        for v in pareto_violations
                    ],
                },
                "status": ("PASS" if not thr_violations and not pareto_violations else "FAIL"),
            },
            indent=2,
        )
    )

    # Console summary.
    print("\n========== Tier B Report ==========")
    print(f"EF sweep: {efs}")
    print(f"Combos:   {COMBO_NAMES}")
    print(f"Threshold violations: {len(thr_violations)}")
    for v in thr_violations:
        print(f"  EF={v.ef} pair={v.pair[0]}-{v.pair[1]} metric={v.metric}: {v.detail}")
    print(f"Pareto violations:    {len(pareto_violations)}")
    for v in pareto_violations:
        print(f"  {v.detail}")
    print(f"Report:   {report_md}")
    print(f"Summary:  {summary_json}")
    print(f"Plot:     {curve_png}")
    print("===================================\n")

    if thr_violations:
        return EXIT_THRESHOLD_FAIL
    if pareto_violations:
        return EXIT_PARETO_FAIL
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
