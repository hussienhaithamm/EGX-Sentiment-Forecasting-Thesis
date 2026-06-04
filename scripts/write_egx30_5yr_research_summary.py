#!/usr/bin/env python3
"""Generate compact thesis-ready EGX30 full-universe result and overfitting reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.sentiment_only_pipeline import compute_direction_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/egx30_5yr_universe_experiments")
DEFAULT_REPORT_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def _markdown_table(rows: Iterable[Dict[str, object]], columns: List[str]) -> str:
    values = list(rows)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in values:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def main() -> int:
    args = parse_args()
    metrics = pd.read_csv(args.output_dir / "metrics" / "direction_metrics.csv")
    predictions = pd.read_csv(args.output_dir / "predictions" / "predictions.csv")
    metrics = _enrich_direction_metrics(metrics, predictions)
    test = metrics[(metrics["split"] == "test") & (metrics["view"] == "bilingual")].copy()
    test = test.sort_values(["sentiment_source", "horizon"])
    display_rows = [
        {
            "Source": row.sentiment_source,
            "Horizon": int(row.horizon),
            "Accuracy": float(row.accuracy),
            "Balanced Accuracy": float(row.balanced_accuracy),
            "F1": float(row.f1),
        }
        for row in test.itertuples(index=False)
    ]
    best = test.sort_values(["balanced_accuracy", "accuracy", "f1"], ascending=False).iloc[0]
    imbalance_rows = []
    for row in test.itertuples(index=False):
        warnings = []
        if getattr(row, "predicted_up_share", 0.5) > 0.90 or getattr(row, "predicted_up_share", 0.5) < 0.10:
            warnings.append("extreme predicted Up share")
        if 0.48 <= float(row.balanced_accuracy) <= 0.52 and float(row.accuracy) >= float(getattr(row, "majority_baseline_accuracy", 1.0)):
            warnings.append("near-random balanced accuracy despite high raw accuracy")
        if warnings:
            imbalance_rows.append(
                {
                    "Source": row.sentiment_source,
                    "View": row.view,
                    "Horizon": int(row.horizon),
                    "Accuracy": float(row.accuracy),
                    "Balanced Accuracy": float(row.balanced_accuracy),
                    "Predicted Up Share": float(getattr(row, "predicted_up_share", 0.0)),
                    "Warning": "; ".join(warnings),
                }
            )

    train_test_rows = []
    for (source, view, horizon), group in predictions.groupby(["sentiment_source", "view", "horizon"], sort=True):
        train = compute_direction_metrics(group[group["split"] == "train"])
        test_metrics = compute_direction_metrics(group[group["split"] == "test"])
        accuracy_gap = float(train["accuracy"] - test_metrics["accuracy"])
        balanced_gap = float(train["balanced_accuracy"] - test_metrics["balanced_accuracy"])
        f1_gap = float(train["f1"] - test_metrics["f1"])
        train_test_rows.append(
            {
                "Source": source,
                "View": view,
                "Horizon": int(horizon),
                "Train Accuracy": float(train["accuracy"]),
                "Test Accuracy": float(test_metrics["accuracy"]),
                "Accuracy Gap": accuracy_gap,
                "Balanced Gap": balanced_gap,
                "F1 Gap": f1_gap,
                "Flag": abs(accuracy_gap) > 0.10 or abs(balanced_gap) > 0.10 or abs(f1_gap) > 0.20,
            }
        )
    flagged = [row for row in train_test_rows if row["Flag"]]
    overfit_rows = sorted(train_test_rows, key=lambda row: abs(float(row["Accuracy Gap"])), reverse=True)

    scoped_counts = {}
    for path in sorted(args.output_dir.glob("*_pit_scoped_articles.csv")):
        scoped_counts[path.name.removesuffix("_pit_scoped_articles.csv")] = len(pd.read_csv(path))
    easy_win_notes = _easy_win_notes(args.report_dir)
    report = f"""# EGX30 Full-Universe Sentiment Forecasting Results

## Scope

- Target: `EGX30` index direction at 1, 3, and 5 trading-day horizons.
- Target price usage: labels only. No price-derived predictor enters the model matrix.
- News universe: survivorship-bias-free point-in-time EGX30 company registry.
- Sentiment sources: `{", ".join(sorted(scoped_counts))}`.
- Strict evaluation: chronological expanding-window walk-forward validation with embargoes.

## Point-in-Time Scoped Articles

{_markdown_table([{"Source": key, "Articles": value} for key, value in scoped_counts.items()], ["Source", "Articles"])}

## Bilingual Test Metrics

{_markdown_table(display_rows, ["Source", "Horizon", "Accuracy", "Balanced Accuracy", "F1"])}

## Best Validity-Gated Bilingual Row

- Source: `{best["sentiment_source"]}`
- Horizon: `{int(best["horizon"])}` trading day(s)
- Accuracy: `{float(best["accuracy"]):.3f}`
- Balanced accuracy: `{float(best["balanced_accuracy"]):.3f}`
- F1: `{float(best["f1"]):.3f}`
- Predicted Up share: `{float(best.get("predicted_up_share", 0.0)):.3f}`
- Actual Up share: `{float(best.get("actual_up_share", 0.0)):.3f}`

## Class-Imbalance Diagnostics

Balanced accuracy remains the primary decision metric because raw accuracy can be partially explained by majority-class behavior.

{_markdown_table(imbalance_rows, ["Source", "View", "Horizon", "Accuracy", "Balanced Accuracy", "Predicted Up Share", "Warning"]) if imbalance_rows else "No bilingual test row crossed the pre-set class-imbalance warning thresholds."}

## Robustness and Easy-Win Validity Extensions

{easy_win_notes}

## Interpretation

The results are promising but limited. Accuracy, balanced accuracy, F1, MCC, actual Up share, and predicted Up share must be interpreted together. Evidence should be described as weak directional signal unless the model beats naive baselines under strict walk-forward validation. The registry, lag, walk-forward, and metric-parity audit is generated separately by `scripts/run_egx30_5yr_validity_checks.py`.
"""
    overfit = f"""# EGX30 Full-Universe Overfitting Report

Rows checked: `{len(train_test_rows)}`

Rows flagged: `{len(flagged)}`

A row is flagged when the absolute train-test accuracy or balanced-accuracy gap exceeds `0.10`, or the absolute F1 gap exceeds `0.20`.

Class-imbalance warnings are reviewed separately because a high raw accuracy can still be weak if balanced accuracy remains close to `0.50` or predictions collapse mostly to one class.

## Largest Accuracy Gaps

{_markdown_table(overfit_rows[:15], ["Source", "View", "Horizon", "Train Accuracy", "Test Accuracy", "Accuracy Gap", "Balanced Gap", "F1 Gap", "Flag"])}

## Reporting Note

Balanced accuracy is prioritized over raw accuracy due to class imbalance. Rows with extreme predicted Up share or near-random balanced accuracy should not be described as strong predictive evidence.
"""
    summary_path = args.output_dir / "research_summary.md"
    overfit_path = args.output_dir / "overfitting_report.md"
    summary_path.write_text(report, encoding="utf-8")
    overfit_path.write_text(overfit, encoding="utf-8")
    update_pipeline_state(
        current_phase="phase_6_research_reporting",
        status="completed",
        completed_phases=["phase_6_research_reporting"],
        artifacts={"research_summary": str(summary_path), "overfitting_report": str(overfit_path)},
    )
    print(summary_path)
    print(overfit_path)
    return 0


def _easy_win_notes(report_dir: Path) -> str:
    paths = {
        "coverage": report_dir / "egx30_coverage_audit.md",
        "baselines": report_dir / "egx30_baseline_comparison.md",
        "abstention": report_dir / "egx30_abstention_experiment.md",
        "sector": report_dir / "egx30_sector_feature_ablation.md",
        "weighting": report_dir / "egx30_weighting_ablation.md",
    }
    lines = [
        "- Coverage audit: company, yearly, and daily coverage reports are generated under `reports/`.",
        "- Baseline comparison: always-Up, always-Down, training-majority, rolling-majority, article-count-only, and shuffled-sentiment sanity baselines are reported.",
        "- Class imbalance: result tables include actual/predicted Up share, majority-baseline accuracy, and delta versus majority behavior.",
        "- Abstention: high-confidence/high-signal results are reported with coverage, covered samples, and abstained samples.",
        "- Sector aggregation: a compact static sector map is used when available and tolerates `UNKNOWN` sectors.",
        "- Weighting: equal-weighted and article-count-weighted sentiment are reported; true historical index weights are marked unavailable when absent.",
    ]
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        lines.append(f"- Missing easy-win report files at summary time: `{', '.join(missing)}`.")
    return "\n".join(lines)


def _enrich_direction_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Backfill newer class-imbalance metrics when reading older saved CSVs."""

    frame = metrics.copy()
    needed = {"mcc", "actual_up_share", "predicted_up_share", "majority_baseline_accuracy", "delta_vs_majority_baseline"}
    if needed.issubset(frame.columns):
        return frame
    rows = []
    for row in frame.to_dict("records"):
        subset = predictions[
            (predictions["horizon"] == row["horizon"])
            & (predictions["sentiment_source"] == row["sentiment_source"])
            & (predictions["view"] == row["view"])
            & (predictions["split"] == row["split"])
        ]
        recomputed = compute_direction_metrics(subset)
        for column in needed:
            row[column] = recomputed[column]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
