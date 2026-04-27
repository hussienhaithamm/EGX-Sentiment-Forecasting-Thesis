"""Run multi-horizon and event-aware sentiment-only EGX evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from egx_direction.event_features import (  # noqa: E402
    aggregate_daily_event_features,
    build_event_feature_panel,
    label_event_articles,
)
from egx_direction.sentiment_only_pipeline import (  # noqa: E402
    LANGUAGE_VIEWS,
    SentimentOnlyPipelineConfig,
    build_time_series_split,
    compute_direction_metrics,
    compute_return_metrics,
    fit_in_sample_direction_predictions,
    fit_in_sample_return_predictions,
    walk_forward_direction_predictions,
    walk_forward_return_predictions,
    _select_best_direction_hyperparameter,
    _select_best_return_hyperparameter,
)
from run_sentiment_only_evaluation_from_dataset import (  # noqa: E402
    aggregate_enhanced_daily_sentiment_features,
    build_enhanced_sentiment_panel,
    load_price_frames,
    load_scored_articles,
)


FEATURE_STRATEGIES = ("all_news", "event_only", "event_weighted")
DEFAULT_THRESHOLD_GRID = tuple(round(value, 3) for value in np.arange(0.30, 0.701, 0.025))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe multi-horizon and event-aware sentiment-only EGX evaluations."
    )
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
        help="Cleaned sentiment-labeled article CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_sentiment_experiments",
        help="Directory for final multi-horizon and event-aware artifacts.",
    )
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"], help="Target EGX tickers.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5], help="Forecast horizons in trading days.")
    parser.add_argument("--price-period", default="10y", help="yfinance fallback period if local prices are missing.")
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[3, 5, 7],
        help="Rolling calendar windows for sentiment features.",
    )
    parser.add_argument(
        "--classification-c-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.5, 1.0],
        help="Validation grid for logistic regression C.",
    )
    parser.add_argument(
        "--regression-alpha-grid",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0],
        help="Validation grid for ridge regression alpha.",
    )
    parser.add_argument(
        "--threshold-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLD_GRID),
        help="Classification thresholds tuned on validation predictions only.",
    )
    parser.add_argument(
        "--refit-every-rows",
        type=int,
        default=20,
        help="Walk-forward model refit cadence. Larger values are faster but still chronological and leakage-safe.",
    )
    return parser.parse_args()


def _score_threshold(metrics: Dict[str, object]) -> Tuple[float, float, float]:
    return (
        float(metrics["balanced_accuracy"]) if not math.isnan(float(metrics["balanced_accuracy"])) else -math.inf,
        float(metrics["f1"]) if not math.isnan(float(metrics["f1"])) else -math.inf,
        float(metrics["accuracy"]) if not math.isnan(float(metrics["accuracy"])) else -math.inf,
    )


def choose_validation_threshold(validation_predictions: pd.DataFrame, threshold_grid: Sequence[float]) -> float:
    """Choose a probability threshold using validation predictions only."""

    if validation_predictions.empty or "predicted_probability" not in validation_predictions.columns:
        return 0.5

    best_threshold = 0.5
    best_score = (-math.inf, -math.inf, -math.inf)
    for threshold in threshold_grid:
        candidate = validation_predictions.copy()
        candidate["predicted_direction"] = (candidate["predicted_probability"].astype(float) >= threshold).astype(int)
        score = _score_threshold(compute_direction_metrics(candidate))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def apply_classification_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = predictions.copy()
    if not frame.empty and "predicted_probability" in frame.columns:
        frame["predicted_direction"] = (frame["predicted_probability"].astype(float) >= threshold).astype(int)
    frame["classification_threshold"] = float(threshold)
    return frame


def _metrics_for_predictions(
    *,
    horizon: int,
    feature_strategy: str,
    ticker: str,
    view: str,
    direction_predictions: pd.DataFrame,
    return_predictions: pd.DataFrame,
    feature_count: int,
    panel_rows: int,
    selected_c: float,
    selected_alpha: float,
    threshold: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    direction_rows: List[Dict[str, object]] = []
    return_rows: List[Dict[str, object]] = []
    for split_label in ("validation", "test"):
        direction_subset = direction_predictions[direction_predictions["split"] == split_label]
        return_subset = return_predictions[return_predictions["split"] == split_label]
        direction_rows.append(
            {
                "horizon": horizon,
                "feature_strategy": feature_strategy,
                "ticker": ticker,
                "view": view,
                "split": split_label,
                "model_family": "logistic_regression",
                "feature_count": feature_count,
                "panel_rows": panel_rows,
                "selected_C": selected_c,
                "classification_threshold": threshold,
                **compute_direction_metrics(direction_subset),
            }
        )
        return_rows.append(
            {
                "horizon": horizon,
                "feature_strategy": feature_strategy,
                "ticker": ticker,
                "view": view,
                "split": split_label,
                "model_family": "ridge_regression",
                "feature_count": feature_count,
                "panel_rows": panel_rows,
                "selected_alpha": selected_alpha,
                **compute_return_metrics(return_subset),
            }
        )
    return direction_rows, return_rows


def _merge_prediction_frames(
    direction_predictions: pd.DataFrame,
    return_predictions: pd.DataFrame,
    *,
    horizon: int,
    feature_strategy: str,
    ticker: str,
    view: str,
    selected_c: float,
    selected_alpha: float,
    threshold: float,
) -> pd.DataFrame:
    common = ["Date", "Ticker", "split", "row_index", "actual_direction", "actual_return"]
    merged = direction_predictions.merge(
        return_predictions,
        on=common,
        how="outer",
        suffixes=("_direction", "_return"),
    )
    merged.insert(0, "horizon", horizon)
    merged.insert(1, "feature_strategy", feature_strategy)
    merged.insert(2, "view", view)
    merged.insert(3, "ticker", ticker)
    merged["selected_C"] = selected_c
    merged["selected_alpha"] = selected_alpha
    merged["classification_threshold"] = threshold
    return merged


def _build_panel(
    *,
    feature_strategy: str,
    ticker: str,
    view: str,
    horizon: int,
    scored_articles: pd.DataFrame,
    price_frame: pd.DataFrame,
    config: SentimentOnlyPipelineConfig,
    feature_dir: Path,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    if feature_strategy == "all_news":
        daily_features = aggregate_enhanced_daily_sentiment_features(scored_articles=scored_articles, view=view)
        ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
        panel, feature_columns = build_enhanced_sentiment_panel(
            price_df=price_frame,
            daily_features=ticker_daily,
            config=config,
        )
    else:
        daily_features = aggregate_daily_event_features(scored_articles=scored_articles, view=view, strategy=feature_strategy)
        ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
        panel, feature_columns = build_event_feature_panel(
            price_df=price_frame,
            daily_features=ticker_daily,
            horizon_days=horizon,
            rolling_windows=config.rolling_windows,
        )

    daily_path = feature_dir / f"h{horizon}_{feature_strategy}_{view}_daily_features.csv"
    if not daily_path.exists():
        daily_features.to_csv(daily_path, index=False)
    return panel, feature_columns, daily_features


def _combined_test_metrics(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame) -> pd.DataFrame:
    direction_test = direction_metrics[direction_metrics["split"] == "test"].copy()
    return_test = return_metrics[return_metrics["split"] == "test"].copy()
    return direction_test.merge(
        return_test,
        on=["horizon", "feature_strategy", "ticker", "view", "split"],
        suffixes=("_direction", "_return"),
    )


def _build_horizon_comparison(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame) -> pd.DataFrame:
    combined = _combined_test_metrics(direction_metrics, return_metrics)
    comparison = combined[
        (combined["feature_strategy"] == "all_news")
        & (combined["view"] == "bilingual")
        & (combined["ticker"] != "ALL")
    ].copy()
    columns = [
        "horizon",
        "ticker",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mae",
        "rmse",
        "sign_accuracy",
        "return_correlation",
        "classification_threshold",
        "selected_C",
        "selected_alpha",
        "rows_direction",
        "rows_return",
    ]
    return comparison[columns].sort_values(["horizon", "ticker"]).reset_index(drop=True)


def _build_event_comparison(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame) -> pd.DataFrame:
    combined = _combined_test_metrics(direction_metrics, return_metrics)
    key = ["horizon", "ticker", "view", "split"]
    baseline = combined[combined["feature_strategy"] == "all_news"].copy()
    rows: List[pd.DataFrame] = []
    for strategy in ("event_only", "event_weighted"):
        candidate = combined[combined["feature_strategy"] == strategy].copy()
        merged = baseline.merge(candidate, on=key, suffixes=("_all_news", f"_{strategy}"))
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mae", "rmse", "sign_accuracy", "return_correlation"):
            merged[f"delta_{metric}_{strategy}"] = merged[f"{metric}_{strategy}"] - merged[f"{metric}_all_news"]
        merged["event_strategy"] = strategy
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["horizon", "ticker", "view", "event_strategy"])


def _add_pooled_rows(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    direction_rows: List[Dict[str, object]] = []
    return_rows: List[Dict[str, object]] = []
    for horizon in sorted(predictions["horizon"].dropna().unique()):
        for feature_strategy in FEATURE_STRATEGIES:
            for view in LANGUAGE_VIEWS:
                for split_label in ("validation", "test"):
                    subset = predictions[
                        (predictions["horizon"] == horizon)
                        & (predictions["feature_strategy"] == feature_strategy)
                        & (predictions["view"] == view)
                        & (predictions["split"] == split_label)
                    ].copy()
                    if subset.empty:
                        continue
                    direction_rows.append(
                        {
                            "horizon": int(horizon),
                            "feature_strategy": feature_strategy,
                            "ticker": "ALL",
                            "view": view,
                            "split": split_label,
                            "model_family": "logistic_regression",
                            "feature_count": math.nan,
                            "panel_rows": math.nan,
                            "selected_C": math.nan,
                            "classification_threshold": math.nan,
                            **compute_direction_metrics(subset),
                        }
                    )
                    return_rows.append(
                        {
                            "horizon": int(horizon),
                            "feature_strategy": feature_strategy,
                            "ticker": "ALL",
                            "view": view,
                            "split": split_label,
                            "model_family": "ridge_regression",
                            "feature_count": math.nan,
                            "panel_rows": math.nan,
                            "selected_alpha": math.nan,
                            **compute_return_metrics(subset.rename(columns={"predicted_return": "predicted_return"})),
                        }
                    )
    return (
        pd.concat([direction_metrics, pd.DataFrame(direction_rows)], ignore_index=True),
        pd.concat([return_metrics, pd.DataFrame(return_rows)], ignore_index=True),
    )


def _best_row(frame: pd.DataFrame, metric: str, maximize: bool = True) -> Dict[str, object]:
    clean = frame[frame["ticker"] != "ALL"].copy()
    clean = clean[pd.notna(clean[metric])]
    if clean.empty:
        return {}
    idx = clean[metric].idxmax() if maximize else clean[metric].idxmin()
    return clean.loc[idx].to_dict()


def _build_summary(
    *,
    output_dir: Path,
    sentiment_labeled_path: Path,
    event_labeled_path: Path,
    horizon_comparison: pd.DataFrame,
    event_comparison: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    return_metrics: pd.DataFrame,
    event_articles: pd.DataFrame,
) -> Dict[str, object]:
    combined = _combined_test_metrics(direction_metrics, return_metrics)
    bilingual = combined[(combined["view"] == "bilingual") & (combined["ticker"] != "ALL")].copy()
    event_help = {}
    if not event_comparison.empty:
        for strategy in ("event_only", "event_weighted"):
            delta_col = f"delta_accuracy_{strategy}"
            if delta_col in event_comparison.columns:
                event_help[strategy] = {
                    "mean_delta_accuracy": float(event_comparison[delta_col].mean()),
                    "best_delta_accuracy": float(event_comparison[delta_col].max()),
                }

    return {
        "sentiment_labeled_path": str(sentiment_labeled_path),
        "output_dir": str(output_dir),
        "event_labeled_articles": str(event_labeled_path),
        "article_counts": event_articles.groupby("Ticker").size().to_dict(),
        "event_article_counts": event_articles.groupby("Ticker")["event_flag"].sum().astype(int).to_dict(),
        "best_overall_accuracy": _best_row(bilingual, "accuracy", True),
        "best_balanced_accuracy": _best_row(bilingual, "balanced_accuracy", True),
        "best_f1": _best_row(bilingual, "f1", True),
        "best_return_sign_accuracy": _best_row(bilingual, "sign_accuracy", True),
        "horizon_comparison_rows": horizon_comparison.to_dict(orient="records"),
        "event_help_summary": event_help,
    }


def run_experiments(
    *,
    sentiment_labeled_path: Path,
    output_dir: Path,
    tickers: Sequence[str],
    horizons: Sequence[int],
    price_period: str,
    rolling_windows: Sequence[int],
    classification_c_grid: Sequence[float],
    regression_alpha_grid: Sequence[float],
    threshold_grid: Sequence[float],
    refit_every_rows: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dirs = {
        "features": output_dir / "features",
        "panels": output_dir / "panels",
        "predictions": output_dir / "predictions",
        "metrics": output_dir / "metrics",
        "comparison": output_dir / "comparison",
    }
    for path in artifact_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    base_config = SentimentOnlyPipelineConfig(
        tickers=tuple(tickers),
        output_dir=output_dir,
        price_period=price_period,
        rolling_windows=tuple(rolling_windows),
        classification_c_grid=tuple(classification_c_grid),
        regression_alpha_grid=tuple(regression_alpha_grid),
        refit_every_rows=refit_every_rows,
    )
    scored_articles = load_scored_articles(sentiment_labeled_path, base_config)
    event_articles = label_event_articles(scored_articles)
    event_labeled_path = output_dir / "event_labeled_articles.csv"
    event_articles.to_csv(event_labeled_path, index=False)

    price_frames = load_price_frames(base_config, output_dir)
    direction_metric_rows: List[Dict[str, object]] = []
    return_metric_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    event_panel_frames: List[pd.DataFrame] = []

    for horizon in horizons:
        config = SentimentOnlyPipelineConfig(
            tickers=tuple(tickers),
            output_dir=output_dir,
            price_period=price_period,
            horizon_days=int(horizon),
            rolling_windows=tuple(rolling_windows),
            classification_c_grid=tuple(classification_c_grid),
            regression_alpha_grid=tuple(regression_alpha_grid),
            refit_every_rows=refit_every_rows,
        )
        for feature_strategy in FEATURE_STRATEGIES:
            strategy_views = LANGUAGE_VIEWS if feature_strategy == "all_news" else ("bilingual",)
            for view in strategy_views:
                print(f"Running horizon={horizon} strategy={feature_strategy} view={view}", flush=True)
                for ticker in tickers:
                    panel, feature_columns, _ = _build_panel(
                        feature_strategy=feature_strategy,
                        ticker=ticker,
                        view=view,
                        horizon=int(horizon),
                        scored_articles=event_articles,
                        price_frame=price_frames[ticker],
                        config=config,
                        feature_dir=artifact_dirs["features"],
                    )
                    if feature_strategy != "all_news" and view == "bilingual":
                        panel_for_export = panel.copy()
                        panel_for_export.insert(0, "horizon", int(horizon))
                        panel_for_export.insert(1, "feature_strategy", feature_strategy)
                        panel_for_export.insert(2, "view", view)
                        metadata_columns = [
                            "horizon",
                            "feature_strategy",
                            "view",
                            "Date",
                            "Ticker",
                            "Close",
                            "target_return",
                            "target_direction",
                            "target_horizon_end_date",
                        ]
                        event_columns = [
                            column
                            for column in panel_for_export.columns
                            if column.startswith("event_") or column.startswith("days_since_event")
                        ]
                        panel_for_export = panel_for_export[metadata_columns + event_columns]
                        event_panel_frames.append(panel_for_export)

                    split = build_time_series_split(
                        n_rows=len(panel),
                        split_config=config.split_config,
                        horizon_days=config.horizon_days,
                    )
                    selected_c, validation_direction = _select_best_direction_hyperparameter(
                        panel,
                        feature_columns,
                        split,
                        config,
                    )
                    selected_alpha, validation_return = _select_best_return_hyperparameter(
                        panel,
                        feature_columns,
                        split,
                        config,
                    )
                    threshold = choose_validation_threshold(validation_direction, threshold_grid)

                    train_direction = fit_in_sample_direction_predictions(
                        panel=panel,
                        feature_columns=feature_columns,
                        train_end=split.train_end,
                        regularization_c=selected_c,
                        random_seed=config.random_seed,
                    )
                    test_direction = walk_forward_direction_predictions(
                        panel=panel,
                        feature_columns=feature_columns,
                        evaluation_indices=split.test_indices,
                        horizon_days=config.horizon_days,
                        regularization_c=selected_c,
                        random_seed=config.random_seed,
                        split_label="test",
                        refit_every_rows=config.refit_every_rows,
                    )
                    direction_predictions = pd.concat(
                        [train_direction, validation_direction, test_direction],
                        ignore_index=True,
                    )
                    direction_predictions = apply_classification_threshold(direction_predictions, threshold)

                    train_return = fit_in_sample_return_predictions(
                        panel=panel,
                        feature_columns=feature_columns,
                        train_end=split.train_end,
                        alpha=selected_alpha,
                    )
                    test_return = walk_forward_return_predictions(
                        panel=panel,
                        feature_columns=feature_columns,
                        evaluation_indices=split.test_indices,
                        horizon_days=config.horizon_days,
                        alpha=selected_alpha,
                        split_label="test",
                        refit_every_rows=config.refit_every_rows,
                    )
                    return_predictions = pd.concat([train_return, validation_return, test_return], ignore_index=True)

                    direction_rows, return_rows = _metrics_for_predictions(
                        horizon=int(horizon),
                        feature_strategy=feature_strategy,
                        ticker=ticker,
                        view=view,
                        direction_predictions=direction_predictions,
                        return_predictions=return_predictions,
                        feature_count=len(feature_columns),
                        panel_rows=len(panel),
                        selected_c=selected_c,
                        selected_alpha=selected_alpha,
                        threshold=threshold,
                    )
                    direction_metric_rows.extend(direction_rows)
                    return_metric_rows.extend(return_rows)
                    prediction_frames.append(
                        _merge_prediction_frames(
                            direction_predictions,
                            return_predictions,
                            horizon=int(horizon),
                            feature_strategy=feature_strategy,
                            ticker=ticker,
                            view=view,
                            selected_c=selected_c,
                            selected_alpha=selected_alpha,
                            threshold=threshold,
                        )
                    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    direction_metrics = pd.DataFrame(direction_metric_rows)
    return_metrics = pd.DataFrame(return_metric_rows)
    direction_metrics, return_metrics = _add_pooled_rows(direction_metrics, return_metrics, predictions)

    horizon_comparison = _build_horizon_comparison(direction_metrics, return_metrics)
    event_model_metrics = _combined_test_metrics(direction_metrics, return_metrics)
    event_vs_all_news = _build_event_comparison(direction_metrics, return_metrics)

    direction_metrics_path = output_dir / "direction_metrics_by_horizon.csv"
    return_metrics_path = output_dir / "return_metrics_by_horizon.csv"
    predictions_path = output_dir / "predictions_by_horizon.csv"
    horizon_comparison_path = output_dir / "horizon_comparison.csv"
    event_model_metrics_path = output_dir / "event_model_metrics.csv"
    event_comparison_path = output_dir / "event_vs_all_news_comparison.csv"
    event_feature_panel_path = output_dir / "event_feature_panel.csv"

    direction_metrics.to_csv(direction_metrics_path, index=False)
    return_metrics.to_csv(return_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    horizon_comparison.to_csv(horizon_comparison_path, index=False)
    event_model_metrics.to_csv(event_model_metrics_path, index=False)
    event_vs_all_news.to_csv(event_comparison_path, index=False)
    if event_panel_frames:
        pd.concat(event_panel_frames, ignore_index=True).to_csv(event_feature_panel_path, index=False)
    else:
        pd.DataFrame().to_csv(event_feature_panel_path, index=False)

    summary = _build_summary(
        output_dir=output_dir,
        sentiment_labeled_path=sentiment_labeled_path,
        event_labeled_path=event_labeled_path,
        horizon_comparison=horizon_comparison,
        event_comparison=event_vs_all_news,
        direction_metrics=direction_metrics,
        return_metrics=return_metrics,
        event_articles=event_articles,
    )
    summary["run_config"] = {
        "tickers": list(tickers),
        "horizons": [int(value) for value in horizons],
        "price_period": price_period,
        "rolling_windows": [int(value) for value in rolling_windows],
        "classification_c_grid": [float(value) for value in classification_c_grid],
        "regression_alpha_grid": [float(value) for value in regression_alpha_grid],
        "threshold_grid": [float(value) for value in threshold_grid],
        "refit_every_rows": int(refit_every_rows),
        "event_views": ["bilingual"],
        "all_news_views": list(LANGUAGE_VIEWS),
    }
    summary["artifacts"] = {
        "horizon_comparison": str(horizon_comparison_path),
        "direction_metrics_by_horizon": str(direction_metrics_path),
        "return_metrics_by_horizon": str(return_metrics_path),
        "predictions_by_horizon": str(predictions_path),
        "summary_by_horizon": str(output_dir / "summary_by_horizon.json"),
        "event_labeled_articles": str(event_labeled_path),
        "event_feature_panel": str(event_feature_panel_path),
        "event_model_metrics": str(event_model_metrics_path),
        "event_vs_all_news_comparison": str(event_comparison_path),
    }
    summary_path = output_dir / "summary_by_horizon.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_experiments(
        sentiment_labeled_path=Path(args.sentiment_labeled_path),
        output_dir=Path(args.output_dir),
        tickers=args.tickers,
        horizons=args.horizons,
        price_period=args.price_period,
        rolling_windows=args.rolling_windows,
        classification_c_grid=args.classification_c_grid,
        regression_alpha_grid=args.regression_alpha_grid,
        threshold_grid=args.threshold_grid,
        refit_every_rows=args.refit_every_rows,
    )
    print("Saved final sentiment experiments to", summary["output_dir"])
    for name, path in summary["artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
