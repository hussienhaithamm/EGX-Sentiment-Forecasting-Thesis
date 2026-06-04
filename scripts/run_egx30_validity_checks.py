"""Validate the EGX30 labels-only sentiment pipeline and extract metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egx_direction.llm_sentiment import attach_llm_sentiment_scores, load_llm_sentiment_cache  # noqa: E402
from egx_direction.market_features import scope_articles_to_market  # noqa: E402
from egx_direction.market_index import DEFAULT_EGX30_PRICE_PATH, EGX30_TICKER, load_egx30_prices  # noqa: E402
from egx_direction.sentiment_only_pipeline import (  # noqa: E402
    LANGUAGE_VIEWS,
    SentimentOnlyPipelineConfig,
    build_time_series_split,
    compute_direction_metrics,
    compute_return_metrics,
)
from run_egx30_sentiment_only_evaluation import build_market_daily_features  # noqa: E402
from run_sentiment_only_evaluation_from_dataset import build_enhanced_sentiment_panel, load_scored_articles  # noqa: E402


EXCLUDED_MODEL_COLUMNS = {
    "Date",
    "Ticker",
    "Close",
    "target_return",
    "target_direction",
    "target_horizon_end_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EGX30 labels-only validity checks and metric extraction.")
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
    )
    parser.add_argument("--target-price-path", default=str(DEFAULT_EGX30_PRICE_PATH))
    parser.add_argument("--llm-cache-path")
    parser.add_argument("--output-dir", default="outputs/egx30_sentiment_only_experiments")
    parser.add_argument("--news-tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 7])
    return parser.parse_args()


def _target_mismatches(panel: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> int:
    close = prices.set_index("Date")["Close"].astype(float)
    mismatches = 0
    for _, row in panel.iterrows():
        date = pd.Timestamp(row["Date"]).normalize()
        end_date = pd.Timestamp(row["target_horizon_end_date"]).normalize()
        expected_return = float(close.loc[end_date] / close.loc[date] - 1.0)
        expected_direction = int(expected_return > 0.0)
        if abs(expected_return - float(row["target_return"])) > 1e-10:
            mismatches += 1
        elif expected_direction != int(row["target_direction"]):
            mismatches += 1
    return mismatches


def _lag_mismatches(panel: pd.DataFrame, daily_features: pd.DataFrame) -> int:
    value_column = "source_company_count"
    observed_column = f"{value_column}_lag1d"
    if daily_features.empty or value_column not in daily_features.columns:
        return 0
    daily = daily_features.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
    lookup = daily.set_index("Date")[value_column].astype(float)
    expected = np.array(
        [
            float(lookup.get(pd.Timestamp(date).normalize() - pd.Timedelta(days=1), 0.0))
            for date in panel["Date"]
        ]
    )
    observed = panel[observed_column].astype(float).to_numpy()
    return int((np.abs(expected - observed) > 1e-10).sum())


def _load_sentiment_sources(
    *,
    sentiment_labeled_path: Path,
    news_tickers: Sequence[str],
    llm_cache_path: Optional[Path],
) -> Dict[str, pd.DataFrame]:
    config = SentimentOnlyPipelineConfig(tickers=tuple(news_tickers))
    scoped = scope_articles_to_market(load_scored_articles(sentiment_labeled_path, config))
    sources = {"base_pipeline": scoped}
    if llm_cache_path is not None:
        sources["llm_zero_shot"] = attach_llm_sentiment_scores(scoped, load_llm_sentiment_cache(llm_cache_path))
    return sources


def _extract_metrics(output_dir: Path) -> pd.DataFrame:
    direction = pd.read_csv(output_dir / "metrics" / "direction_metrics_by_horizon.csv")
    returns = pd.read_csv(output_dir / "metrics" / "return_metrics_by_horizon.csv")
    if "sentiment_source" not in direction.columns:
        direction["sentiment_source"] = "base_pipeline"
    if "sentiment_source" not in returns.columns:
        returns["sentiment_source"] = "base_pipeline"
    keys = ["horizon", "feature_strategy", "sentiment_source", "ticker", "view", "split"]
    extracted = direction.merge(
        returns[keys + ["mae", "rmse", "sign_accuracy", "return_correlation"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return extracted.sort_values(keys).reset_index(drop=True)


def _prediction_checks(output_dir: Path) -> Dict[str, object]:
    predictions = pd.read_csv(output_dir / "predictions" / "predictions_by_horizon.csv")
    if "sentiment_source" not in predictions.columns:
        predictions["sentiment_source"] = "base_pipeline"
    evaluation = predictions[predictions["split"].isin(["validation", "test"])].copy()
    evaluation["max_allowed_history_rows"] = evaluation["row_index"] - evaluation["horizon"] + 1
    direction_violations = int(
        (evaluation["history_rows_direction"] > evaluation["max_allowed_history_rows"]).sum()
    )
    return_violations = int(
        (evaluation["history_rows_return"] > evaluation["max_allowed_history_rows"]).sum()
    )
    metric_rows: List[Dict[str, object]] = []
    group_columns = ["horizon", "feature_strategy", "sentiment_source", "ticker", "view", "split"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row.update({f"direction_{key}": value for key, value in compute_direction_metrics(group).items()})
        row.update({f"return_{key}": value for key, value in compute_return_metrics(group).items()})
        metric_rows.append(row)
    recomputed = pd.DataFrame(metric_rows)
    recomputed.to_csv(output_dir / "validity" / "recomputed_metrics_from_predictions.csv", index=False)
    return {
        "evaluation_rows_checked": int(len(evaluation)),
        "direction_history_violations": direction_violations,
        "return_history_violations": return_violations,
        "rule": "history_rows <= row_index - horizon + 1 for validation/test rows",
    }


def _target_provenance_checks(target_price_path: Path, prices: pd.DataFrame) -> Dict[str, object]:
    metadata_path = target_price_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {"status": "FAIL", "reason": f"Missing metadata sidecar: {metadata_path}"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_sha256 = hashlib.sha256(target_price_path.read_bytes()).hexdigest()
    expected = {
        "target": EGX30_TICKER,
        "usage": "target_construction_only",
        "rows": int(len(prices)),
        "first_date": prices["Date"].min().date().isoformat(),
        "last_date": prices["Date"].max().date().isoformat(),
        "sha256": actual_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "metadata_path": str(metadata_path),
        "mismatches": mismatches,
    }


def _metric_parity_mismatches(output_dir: Path, extracted: pd.DataFrame) -> List[str]:
    recomputed = pd.read_csv(output_dir / "validity" / "recomputed_metrics_from_predictions.csv")
    keys = ["horizon", "feature_strategy", "sentiment_source", "ticker", "view", "split"]
    recomputed = recomputed[recomputed["split"].isin(["validation", "test"])].copy()
    columns = {
        "accuracy": "direction_accuracy",
        "balanced_accuracy": "direction_balanced_accuracy",
        "f1": "direction_f1",
        "mae": "return_mae",
        "rmse": "return_rmse",
        "sign_accuracy": "return_sign_accuracy",
        "return_correlation": "return_return_correlation",
    }
    merged = extracted.merge(
        recomputed[keys + list(columns.values())],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    mismatches: List[str] = []
    for _, row in merged.iterrows():
        identifier = "/".join(str(row[key]) for key in keys)
        if row["_merge"] != "both":
            mismatches.append(f"{identifier}: metric row is {row['_merge']}")
            continue
        for exported_column, recomputed_column in columns.items():
            exported = float(row[exported_column])
            expected = float(row[recomputed_column])
            if not np.isclose(exported, expected, equal_nan=True, atol=1e-12):
                mismatches.append(
                    f"{identifier}: {exported_column} exported={exported} recomputed={expected}"
                )
    return mismatches


def _write_markdown(report: Dict[str, object], path: Path) -> None:
    lines = [
        "# EGX30 Validity Report",
        "",
        f"Code and data validity: **{report['validity_status']}**",
        f"Experiment completeness: **{report['experiment_completeness']}**",
        "",
        "## Target",
        "",
        f"- Target: `{report['target']}` directional movement.",
        "- EGX30 closes are used only to construct future labels.",
        f"- Price rows: `{report['target_price_rows']}`.",
        f"- Timeline: `{report['target_price_start']}` to `{report['target_price_end']}`.",
        "",
        "## Leakage Checks",
        "",
        f"- Panels checked: `{len(report['panel_checks'])}`.",
        f"- Walk-forward rows checked: `{report['walk_forward_history_check']['evaluation_rows_checked']}`.",
        f"- Direction history violations: `{report['walk_forward_history_check']['direction_history_violations']}`.",
        f"- Return history violations: `{report['walk_forward_history_check']['return_history_violations']}`.",
        f"- Panel violations: `{len(report['violations'])}`.",
        f"- Exported metric parity mismatches: `{len(report['metric_parity_mismatches'])}`.",
        f"- Pinned target provenance: `{report['target_provenance']['status']}`.",
        "",
        "## LLM Baseline",
        "",
        f"- Status: `{report['llm_baseline_status']}`.",
    ]
    if report["violations"]:
        lines.extend(["", "## Violations", ""])
        lines.extend(f"- {item}" for item in report["violations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(
    *,
    sentiment_labeled_path: Path,
    target_price_path: Path,
    llm_cache_path: Optional[Path],
    output_dir: Path,
    news_tickers: Sequence[str],
    horizons: Sequence[int],
    rolling_windows: Sequence[int],
) -> Dict[str, object]:
    validity_dir = output_dir / "validity"
    validity_dir.mkdir(parents=True, exist_ok=True)
    prices = load_egx30_prices(target_price_path)
    target_provenance = _target_provenance_checks(target_price_path, prices)
    sources = _load_sentiment_sources(
        sentiment_labeled_path=sentiment_labeled_path,
        news_tickers=news_tickers,
        llm_cache_path=llm_cache_path,
    )
    violations: List[str] = []
    panel_checks: List[Dict[str, object]] = []
    for sentiment_source, source_articles in sources.items():
        for horizon in horizons:
            config = SentimentOnlyPipelineConfig(
                tickers=(EGX30_TICKER,),
                horizon_days=int(horizon),
                rolling_windows=tuple(rolling_windows),
            )
            for view in LANGUAGE_VIEWS:
                daily = build_market_daily_features(source_articles, view)
                panel, feature_columns = build_enhanced_sentiment_panel(prices, daily, config)
                leaked = sorted(set(feature_columns) & EXCLUDED_MODEL_COLUMNS)
                target_mismatches = _target_mismatches(panel, prices, int(horizon))
                lag_mismatches = _lag_mismatches(panel, daily)
                split = build_time_series_split(len(panel), config.split_config, int(horizon))
                embargo = max(config.split_config.embargo_rows, int(horizon))
                split_embargo_ok = (
                    split.validation_start >= split.train_end + embargo
                    and split.test_start >= split.validation_end + embargo
                )
                check_id = f"{sentiment_source}/h{horizon}/{view}"
                if leaked:
                    violations.append(f"{check_id}: leaked model columns {leaked}")
                if target_mismatches:
                    violations.append(f"{check_id}: {target_mismatches} target mismatches")
                if lag_mismatches:
                    violations.append(f"{check_id}: {lag_mismatches} one-day lag mismatches")
                if not split_embargo_ok:
                    violations.append(f"{check_id}: split embargo failed")
                panel_checks.append(
                    {
                        "sentiment_source": sentiment_source,
                        "horizon": int(horizon),
                        "view": view,
                        "panel_rows": int(len(panel)),
                        "feature_count": int(len(feature_columns)),
                        "leaked_model_columns": leaked,
                        "target_mismatches": int(target_mismatches),
                        "lag_mismatches": int(lag_mismatches),
                        "split_embargo_ok": split_embargo_ok,
                    }
                )

    walk_forward = _prediction_checks(output_dir)
    if walk_forward["direction_history_violations"]:
        violations.append("Direction walk-forward predictions exceed allowed history.")
    if walk_forward["return_history_violations"]:
        violations.append("Return walk-forward predictions exceed allowed history.")
    metrics = _extract_metrics(output_dir)
    metrics.to_csv(validity_dir / "egx30_metric_extract.csv", index=False)
    metrics[metrics["split"] == "test"].to_csv(validity_dir / "egx30_test_metrics.csv", index=False)
    metric_parity_mismatches = _metric_parity_mismatches(output_dir, metrics)
    if target_provenance["status"] != "PASS":
        violations.append("Pinned EGX30 target provenance check failed.")
    violations.extend(metric_parity_mismatches)

    llm_present = "llm_zero_shot" in sources
    report = {
        "target": EGX30_TICKER,
        "target_usage": "labels_only",
        "target_price_rows": int(len(prices)),
        "target_price_start": prices["Date"].min().date().isoformat(),
        "target_price_end": prices["Date"].max().date().isoformat(),
        "target_price_columns": prices.columns.tolist(),
        "target_provenance": target_provenance,
        "sentiment_sources_checked": list(sources),
        "llm_baseline_status": "complete" if llm_present else "blocked_missing_complete_cache",
        "experiment_completeness": "COMPLETE" if llm_present else "BLOCKED",
        "validity_status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "metric_parity_mismatches": metric_parity_mismatches,
        "panel_checks": panel_checks,
        "walk_forward_history_check": walk_forward,
    }
    (validity_dir / "egx30_validity_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_markdown(report, validity_dir / "egx30_validity_report.md")
    return report


def main() -> None:
    args = parse_args()
    report = run_checks(
        sentiment_labeled_path=Path(args.sentiment_labeled_path),
        target_price_path=Path(args.target_price_path),
        llm_cache_path=Path(args.llm_cache_path) if args.llm_cache_path else None,
        output_dir=Path(args.output_dir),
        news_tickers=args.news_tickers,
        horizons=args.horizons,
        rolling_windows=args.rolling_windows,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
