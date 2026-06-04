"""Run the strict walk-forward EGX30 sentiment-only baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egx_direction.market_features import (  # noqa: E402
    MARKET_SCOPE_FEATURE_COLUMNS,
    aggregate_daily_market_scope_features,
    scope_articles_to_market,
)
from egx_direction.llm_sentiment import (  # noqa: E402
    attach_llm_sentiment_scores,
    load_llm_sentiment_cache,
)
from egx_direction.market_index import (  # noqa: E402
    DEFAULT_EGX30_PRICE_PATH,
    EGX30_TICKER,
    load_egx30_prices,
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
from run_multi_horizon_event_evaluation import (  # noqa: E402
    DEFAULT_THRESHOLD_GRID,
    apply_classification_threshold,
    choose_validation_threshold,
)
from run_sentiment_only_evaluation_from_dataset import (  # noqa: E402
    aggregate_enhanced_daily_sentiment_features,
    build_enhanced_sentiment_panel,
    load_scored_articles,
)


EXCLUDED_MODEL_COLUMNS = {
    "Date",
    "Ticker",
    "Close",
    "target_return",
    "target_direction",
    "target_horizon_end_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EGX30 market-mood sentiment-only benchmark.")
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
    )
    parser.add_argument("--target-price-path", default=str(DEFAULT_EGX30_PRICE_PATH))
    parser.add_argument(
        "--llm-cache-path",
        help="Optional complete article-level GPT zero-shot sentiment cache for the modern baseline.",
    )
    parser.add_argument("--output-dir", default="outputs/egx30_sentiment_only_experiments")
    parser.add_argument("--news-tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--classification-c-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument("--regression-alpha-grid", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=list(DEFAULT_THRESHOLD_GRID))
    parser.add_argument("--refit-every-rows", type=int, default=20)
    return parser.parse_args()


def build_market_daily_features(scoped_articles: pd.DataFrame, view: str) -> pd.DataFrame:
    """Combine weighted sentiment aggregates with firm/macro source counts."""

    sentiment = aggregate_enhanced_daily_sentiment_features(scoped_articles, view)
    scope = aggregate_daily_market_scope_features(scoped_articles, view)
    if sentiment.empty:
        return scope
    return sentiment.merge(scope, on=["Ticker", "Date"], how="left").fillna(
        {column: 0.0 for column in MARKET_SCOPE_FEATURE_COLUMNS}
    )


def _merge_predictions(
    direction: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    horizon: int,
    view: str,
    selected_c: float,
    selected_alpha: float,
    threshold: float,
    sentiment_source: str,
) -> pd.DataFrame:
    common = ["Date", "Ticker", "split", "row_index", "actual_direction", "actual_return"]
    merged = direction.merge(returns, on=common, how="outer", suffixes=("_direction", "_return"))
    merged.insert(0, "horizon", int(horizon))
    merged.insert(1, "feature_strategy", "market_mood")
    merged.insert(2, "view", view)
    merged.insert(3, "ticker", EGX30_TICKER)
    merged.insert(4, "sentiment_source", sentiment_source)
    merged["selected_C"] = float(selected_c)
    merged["selected_alpha"] = float(selected_alpha)
    merged["classification_threshold"] = float(threshold)
    return merged


def run_experiments(
    *,
    sentiment_labeled_path: Path,
    target_price_path: Path,
    llm_cache_path: Optional[Path],
    output_dir: Path,
    news_tickers: Sequence[str],
    horizons: Sequence[int],
    rolling_windows: Sequence[int],
    classification_c_grid: Sequence[float],
    regression_alpha_grid: Sequence[float],
    threshold_grid: Sequence[float],
    refit_every_rows: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("features", "panels", "predictions", "metrics", "prices"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    news_config = SentimentOnlyPipelineConfig(tickers=tuple(news_tickers))
    scored_articles = load_scored_articles(sentiment_labeled_path, news_config)
    scoped_articles = scope_articles_to_market(scored_articles)
    scoped_articles.to_csv(output_dir / "market_scoped_articles.csv", index=False)
    sentiment_sources = {"base_pipeline": scoped_articles}
    if llm_cache_path is not None:
        llm_cache = load_llm_sentiment_cache(llm_cache_path)
        sentiment_sources["llm_zero_shot"] = attach_llm_sentiment_scores(scoped_articles, llm_cache)
    target_prices = load_egx30_prices(target_price_path)
    target_prices.to_csv(output_dir / "prices" / "EGX30_price_history.csv", index=False)

    direction_metric_rows: List[Dict[str, object]] = []
    return_metric_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for horizon in horizons:
        config = SentimentOnlyPipelineConfig(
            tickers=(EGX30_TICKER,),
            output_dir=output_dir,
            horizon_days=int(horizon),
            rolling_windows=tuple(rolling_windows),
            classification_c_grid=tuple(classification_c_grid),
            regression_alpha_grid=tuple(regression_alpha_grid),
            refit_every_rows=int(refit_every_rows),
        )
        for sentiment_source, source_articles in sentiment_sources.items():
            for view in LANGUAGE_VIEWS:
                print(
                    f"Running target={EGX30_TICKER} horizon={horizon} "
                    f"sentiment_source={sentiment_source} view={view}",
                    flush=True,
                )
                daily_features = build_market_daily_features(source_articles, view)
                artifact_prefix = f"h{horizon}_{sentiment_source}_market_mood_{view}"
                daily_features.to_csv(output_dir / "features" / f"{artifact_prefix}_daily_features.csv", index=False)
                panel, feature_columns = build_enhanced_sentiment_panel(target_prices, daily_features, config)
                leaked = sorted(set(feature_columns) & EXCLUDED_MODEL_COLUMNS)
                if leaked:
                    raise RuntimeError(f"Target-only columns leaked into model matrix: {leaked}")
                panel.to_csv(output_dir / "panels" / f"{artifact_prefix}_panel.csv", index=False)

                split = build_time_series_split(len(panel), config.split_config, config.horizon_days)
                selected_c, validation_direction = _select_best_direction_hyperparameter(
                    panel, feature_columns, split, config
                )
                selected_alpha, validation_return = _select_best_return_hyperparameter(
                    panel, feature_columns, split, config
                )
                threshold = choose_validation_threshold(validation_direction, threshold_grid)

                train_direction = fit_in_sample_direction_predictions(
                    panel, feature_columns, split.train_end, selected_c, config.random_seed
                )
                test_direction = walk_forward_direction_predictions(
                    panel,
                    feature_columns,
                    split.test_indices,
                    config.horizon_days,
                    selected_c,
                    config.random_seed,
                    split_label="test",
                    refit_every_rows=config.refit_every_rows,
                )
                direction = apply_classification_threshold(
                    pd.concat([train_direction, validation_direction, test_direction], ignore_index=True),
                    threshold,
                )
                train_return = fit_in_sample_return_predictions(panel, feature_columns, split.train_end, selected_alpha)
                test_return = walk_forward_return_predictions(
                    panel,
                    feature_columns,
                    split.test_indices,
                    config.horizon_days,
                    selected_alpha,
                    split_label="test",
                    refit_every_rows=config.refit_every_rows,
                )
                returns = pd.concat([train_return, validation_return, test_return], ignore_index=True)

                for split_label in ("validation", "test"):
                    direction_metric_rows.append(
                        {
                            "horizon": int(horizon),
                            "feature_strategy": "market_mood",
                            "sentiment_source": sentiment_source,
                            "ticker": EGX30_TICKER,
                            "view": view,
                            "split": split_label,
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                            "selected_C": float(selected_c),
                            "classification_threshold": float(threshold),
                            **compute_direction_metrics(direction[direction["split"] == split_label]),
                        }
                    )
                    return_metric_rows.append(
                        {
                            "horizon": int(horizon),
                            "feature_strategy": "market_mood",
                            "sentiment_source": sentiment_source,
                            "ticker": EGX30_TICKER,
                            "view": view,
                            "split": split_label,
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                            "selected_alpha": float(selected_alpha),
                            **compute_return_metrics(returns[returns["split"] == split_label]),
                        }
                    )
                prediction_frames.append(
                    _merge_predictions(
                        direction,
                        returns,
                        horizon=int(horizon),
                        view=view,
                        selected_c=selected_c,
                        selected_alpha=selected_alpha,
                        threshold=threshold,
                        sentiment_source=sentiment_source,
                    )
                )

    direction_metrics = pd.DataFrame(direction_metric_rows)
    return_metrics = pd.DataFrame(return_metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    direction_metrics.to_csv(output_dir / "metrics" / "direction_metrics_by_horizon.csv", index=False)
    return_metrics.to_csv(output_dir / "metrics" / "return_metrics_by_horizon.csv", index=False)
    predictions.to_csv(output_dir / "predictions" / "predictions_by_horizon.csv", index=False)
    summary = {
        "target": EGX30_TICKER,
        "target_usage": "labels_only",
        "target_price_path": str(target_price_path),
        "llm_cache_path": str(llm_cache_path) if llm_cache_path is not None else None,
        "sentiment_sources": list(sentiment_sources),
        "news_tickers": list(news_tickers),
        "scored_articles_before_market_dedup": int(len(scored_articles)),
        "market_articles_after_dedup": int(len(scoped_articles)),
        "horizons": [int(value) for value in horizons],
        "rolling_windows": [int(value) for value in rolling_windows],
        "direction_metrics": direction_metrics.to_dict(orient="records"),
        "return_metrics": return_metrics.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_experiments(
        sentiment_labeled_path=Path(args.sentiment_labeled_path),
        target_price_path=Path(args.target_price_path),
        llm_cache_path=Path(args.llm_cache_path) if args.llm_cache_path else None,
        output_dir=Path(args.output_dir),
        news_tickers=args.news_tickers,
        horizons=args.horizons,
        rolling_windows=args.rolling_windows,
        classification_c_grid=args.classification_c_grid,
        regression_alpha_grid=args.regression_alpha_grid,
        threshold_grid=args.threshold_grid,
        refit_every_rows=args.refit_every_rows,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
