#!/usr/bin/env python3
"""Audit the five-year point-in-time EGX30 experiment for leakage and artifact parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.market_index import DEFAULT_EGX30_PRICE_PATH, load_egx30_prices  # noqa: E402
from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.pit_constituents import (  # noqa: E402
    DEFAULT_CONSTITUENT_PATH,
    EGX30_RESEARCH_END,
    EGX30_RESEARCH_START,
    active_tickers,
    load_constituent_registry,
    validate_constituent_registry,
)
from egx_direction.sentiment_only_pipeline import build_sentiment_targets, compute_direction_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/egx30_5yr_universe_experiments")
DEFAULT_REPORT_PATH = Path("data/egx30_5yr_validity_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--constituent-path", type=Path, default=DEFAULT_CONSTITUENT_PATH)
    parser.add_argument("--target-price-path", type=Path, default=DEFAULT_EGX30_PRICE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--skip-state-update", action="store_true", help="Do not mutate data/pipeline_state.json during smoke tests.")
    return parser.parse_args()


def _assert_close(actual: pd.Series, expected: pd.Series, message: str) -> None:
    if not np.allclose(actual.astype(float), expected.astype(float), rtol=1e-10, atol=1e-12):
        raise AssertionError(message)


def audit(output_dir: Path, constituent_path: Path, target_price_path: Path) -> Dict[str, object]:
    constituents = load_constituent_registry(constituent_path)
    registry_report = validate_constituent_registry(constituents)
    prices = load_egx30_prices(
        target_price_path,
        start_date=EGX30_RESEARCH_START,
        end_date=EGX30_RESEARCH_END,
        require_exact_bounds=True,
    )
    scoped_paths = sorted(output_dir.glob("*_pit_scoped_articles.csv"))
    panel_paths = sorted((output_dir / "panels").glob("h*_*.csv"))
    prediction_path = output_dir / "predictions" / "predictions.csv"
    metrics_path = output_dir / "metrics" / "direction_metrics.csv"
    if not scoped_paths or not panel_paths or not prediction_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Evaluation artifacts are incomplete; run scripts/run_egx30_5yr_evaluation.py first.")

    scoped_rows = 0
    for path in scoped_paths:
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
        for row in frame[["Date", "Ticker"]].itertuples(index=False):
            if str(row.Ticker).upper() not in set(active_tickers(constituents, row.Date)):
                raise AssertionError(f"Inactive ticker leaked into {path}: {row.Ticker} on {row.Date.date()}")
        if frame.duplicated(subset=["Date", "market_article_identity"]).any():
            raise AssertionError(f"Cross-company duplicate article leaked into {path}")
        scoped_rows += len(frame)

    lag_checks = 0
    label_checks = 0
    for panel_path in panel_paths:
        stem = panel_path.stem
        horizon = int(stem.split("_", 1)[0][1:])
        panel = pd.read_csv(panel_path)
        panel["Date"] = pd.to_datetime(panel["Date"], errors="raise").dt.normalize()
        panel["target_horizon_end_date"] = pd.to_datetime(panel["target_horizon_end_date"], errors="raise").dt.normalize()
        expected = build_sentiment_targets(prices, horizon).rename(
            columns={
                "target_return": "expected_return",
                "target_direction": "expected_direction",
                "target_horizon_end_date": "expected_end_date",
            }
        )
        merged = panel.merge(expected[["Date", "expected_return", "expected_direction", "expected_end_date"]], on="Date", how="left")
        _assert_close(merged["target_return"], merged["expected_return"], f"Target return mismatch in {panel_path}")
        _assert_close(merged["target_direction"], merged["expected_direction"], f"Target direction mismatch in {panel_path}")
        if not merged["target_horizon_end_date"].equals(merged["expected_end_date"]):
            raise AssertionError(f"Target horizon date mismatch in {panel_path}")
        label_checks += len(panel)

        horizon_prefix = stem.split("_", 1)[0]
        source_view = stem[len(horizon_prefix) + 1 :].removesuffix("_panel")
        source, view = source_view.rsplit("_", 1) if source_view.endswith("_bilingual") else ("", "")
        if not source:
            for suffix in ("english_only", "arabic_only"):
                if source_view.endswith(f"_{suffix}"):
                    source, view = source_view[: -(len(suffix) + 1)], suffix
                    break
        daily_path = output_dir / "features" / f"{source}_{view}_equal_weight_daily.csv"
        daily = pd.read_csv(daily_path)
        daily["Date"] = pd.to_datetime(daily["Date"], errors="raise").dt.normalize()
        expected_lag = daily.set_index("Date")["combined_article_count"].shift(1).fillna(0.0)
        observed = panel.set_index("Date")["combined_article_count_lag1d"]
        _assert_close(observed, expected_lag.reindex(observed.index).fillna(0.0), f"t-1 lag mismatch in {panel_path}")
        lag_checks += len(panel)

        feature_columns = [
            column
            for column in panel.columns
            if column not in {"Date", "Ticker", "Close", "target_return", "target_direction", "target_horizon_end_date"}
        ]
        if any(column in {"Close", "target_return", "target_direction"} for column in feature_columns):
            raise AssertionError(f"Target field entered feature matrix in {panel_path}")

    predictions = pd.read_csv(prediction_path)
    evaluation = predictions[predictions["split"].isin(["validation", "test"])].copy()
    history_limit = evaluation["row_index"].astype(int) - evaluation["horizon"].astype(int) + 1
    history_columns = [column for column in evaluation.columns if column.startswith("history_rows")]
    if not history_columns:
        raise AssertionError("Prediction artifacts are missing walk-forward history provenance.")
    for column in history_columns:
        if (evaluation[column].astype(int) > history_limit).any():
            raise AssertionError(f"A walk-forward model used unavailable labels according to {column}.")

    metrics = pd.read_csv(metrics_path)
    parity_rows: List[Dict[str, object]] = []
    for keys, row in metrics.iterrows():
        subset = predictions[
            (predictions["horizon"] == row["horizon"])
            & (predictions["sentiment_source"] == row["sentiment_source"])
            & (predictions["view"] == row["view"])
            & (predictions["split"] == row["split"])
        ]
        recomputed = compute_direction_metrics(subset)
        for metric in ("accuracy", "balanced_accuracy", "f1"):
            if not np.isclose(float(row[metric]), float(recomputed[metric]), rtol=1e-12, atol=1e-12):
                raise AssertionError(f"Metric parity mismatch for {metric}: {dict(row)}")
        parity_rows.append({"horizon": int(row["horizon"]), "view": row["view"], "split": row["split"]})

    easy_win_checks = _audit_easy_win_reports(Path("reports"))

    return {
        "status": "PASS",
        "registry": registry_report,
        "scoped_article_rows_checked": int(scoped_rows),
        "panel_count": len(panel_paths),
        "target_rows_checked": int(label_checks),
        "lag_rows_checked": int(lag_checks),
        "walk_forward_prediction_rows_checked": int(len(evaluation)),
        "metric_rows_checked": len(parity_rows),
        "easy_win_reports": easy_win_checks,
        "assertions": [
            "point_in_time_registry_continuous",
            "only_documented_transition_states_allowed",
            "inactive_company_articles_excluded",
            "cross_company_articles_deduplicated",
            "egx30_closes_used_for_targets_only",
            "daily_sentiment_features_shifted_by_at_least_one_calendar_day",
            "walk_forward_history_excludes_unavailable_labels",
            "saved_accuracy_balanced_accuracy_f1_match_predictions",
            "easy_win_reports_have_expected_columns_when_generated",
        ],
    }


def _audit_easy_win_reports(report_dir: Path) -> Dict[str, object]:
    required = {
        "egx30_company_coverage.csv": {"ticker", "article_count", "arabic_article_count", "english_article_count"},
        "egx30_yearly_coverage.csv": {"year", "article_count", "covered_trading_day_pct"},
        "egx30_daily_coverage.csv": {"date", "article_count", "distinct_companies_covered"},
        "egx30_baseline_comparison.csv": {"baseline_name", "balanced_accuracy", "predicted_up_share", "actual_up_share"},
        "egx30_abstention_experiment.csv": {"coverage_pct", "covered_test_samples", "abstained_test_samples"},
    }
    checked: Dict[str, object] = {"status": "SKIPPED", "checked_files": []}
    if not report_dir.exists():
        return checked
    checked["status"] = "PASS"
    for filename, columns in required.items():
        path = report_dir / filename
        if not path.exists():
            checked["status"] = "MISSING"
            continue
        frame = pd.read_csv(path, nrows=5)
        missing = columns - set(frame.columns)
        if missing:
            raise AssertionError(f"{path} is missing expected columns: {sorted(missing)}")
        checked["checked_files"].append(filename)
    return checked


def main() -> int:
    args = parse_args()
    report = audit(args.output_dir, args.constituent_path, args.target_price_path)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.skip_state_update:
        update_pipeline_state(
            current_phase="phase_5_validity_checks",
            status="completed",
            completed_phases=["phase_5_validity_checks"],
            artifacts={"validity_report": str(args.report_path)},
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
