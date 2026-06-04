#!/usr/bin/env python3
"""Run strict walk-forward EGX30 sentiment-only forecasting on the full historical universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egx_direction.llm_sentiment import attach_llm_sentiment_scores, load_llm_sentiment_cache  # noqa: E402
from egx_direction.market_index import DEFAULT_EGX30_PRICE_PATH, EGX30_TICKER, load_egx30_prices  # noqa: E402
from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.pit_constituents import (  # noqa: E402
    DEFAULT_CONSTITUENT_PATH,
    EGX30_RESEARCH_END,
    EGX30_RESEARCH_START,
    load_constituent_registry,
    validate_constituent_registry,
)
from egx_direction.pit_market_features import (  # noqa: E402
    aggregate_equal_weight_market_daily,
    scope_articles_to_pit_market,
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
)


DEFAULT_PROCESSED_NEWS = Path("data/processed_news_5yr.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/egx30_5yr_universe_experiments")
DEFAULT_METRICS_PATH = Path("data/egx30_5yr_universe_metrics.csv")
BANNED_FEATURE_TOKENS = ("close", "open", "high", "low", "volume", "price", "target", "return")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-news-path", type=Path, default=DEFAULT_PROCESSED_NEWS)
    parser.add_argument("--constituent-path", type=Path, default=DEFAULT_CONSTITUENT_PATH)
    parser.add_argument("--target-price-path", type=Path, default=DEFAULT_EGX30_PRICE_PATH)
    parser.add_argument("--llm-cache-path", type=Path)
    parser.add_argument("--camelbert-processed-news-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--classification-c-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument("--regression-alpha-grid", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=list(DEFAULT_THRESHOLD_GRID))
    parser.add_argument("--refit-every-rows", type=int, default=20)
    parser.add_argument("--skip-state-update", action="store_true", help="Do not mutate data/pipeline_state.json during smoke tests.")
    return parser.parse_args()


def load_processed_articles(path: Path) -> pd.DataFrame:
    """Load scored articles while restoring timestamp columns serialized in CSV."""

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No processed articles found: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    return frame.dropna(subset=["Ticker", "Date", "published_at"]).reset_index(drop=True)


def build_market_daily_features(
    source_articles: pd.DataFrame,
    constituents: pd.DataFrame,
    view: str,
) -> pd.DataFrame:
    """Aggregate firm mood first, then average active firms equally."""

    company_daily = aggregate_enhanced_daily_sentiment_features(source_articles, view)
    return aggregate_equal_weight_market_daily(company_daily, constituents)


def _merge_predictions(
    direction: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    horizon: int,
    view: str,
    source: str,
    selected_c: float,
    selected_alpha: float,
    threshold: float,
) -> pd.DataFrame:
    common = ["Date", "Ticker", "split", "row_index", "actual_direction", "actual_return"]
    merged = direction.merge(returns, on=common, how="outer", suffixes=("_direction", "_return"))
    merged.insert(0, "horizon", int(horizon))
    merged.insert(1, "sentiment_source", source)
    merged.insert(2, "view", view)
    merged["selected_C"] = float(selected_c)
    merged["selected_alpha"] = float(selected_alpha)
    merged["classification_threshold"] = float(threshold)
    return merged


def run_experiments(
    *,
    processed_news_path: Path,
    constituent_path: Path,
    target_price_path: Path,
    llm_cache_path: Optional[Path],
    camelbert_processed_news_path: Optional[Path],
    output_dir: Path,
    metrics_path: Path,
    horizons: Sequence[int],
    rolling_windows: Sequence[int],
    classification_c_grid: Sequence[float],
    regression_alpha_grid: Sequence[float],
    threshold_grid: Sequence[float],
    refit_every_rows: int,
    update_state: bool = True,
) -> Dict[str, object]:
    if update_state:
        update_pipeline_state(current_phase="phase_4_strict_walk_forward_evaluation", status="in_progress")
    for child in ("features", "panels", "predictions", "metrics", "prices"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    constituents = load_constituent_registry(constituent_path)
    validation = validate_constituent_registry(constituents)
    scored = load_processed_articles(processed_news_path)
    sentiment_sources = {"finbert_arabert": scope_articles_to_pit_market(scored, constituents)}
    if camelbert_processed_news_path is not None:
        sentiment_sources["finbert_camelbert"] = scope_articles_to_pit_market(
            load_processed_articles(camelbert_processed_news_path),
            constituents,
        )
    if llm_cache_path is not None:
        llm_scored = attach_llm_sentiment_scores(scored, load_llm_sentiment_cache(llm_cache_path))
        sentiment_sources["gpt_zero_shot"] = scope_articles_to_pit_market(llm_scored, constituents)
    for source, articles in sentiment_sources.items():
        articles.to_csv(output_dir / f"{source}_pit_scoped_articles.csv", index=False)

    prices = load_egx30_prices(
        target_price_path,
        start_date=EGX30_RESEARCH_START,
        end_date=EGX30_RESEARCH_END,
        require_exact_bounds=True,
    )
    prices.to_csv(output_dir / "prices" / "EGX30_target_only_closes.csv", index=False)
    metric_rows: List[Dict[str, object]] = []
    return_rows: List[Dict[str, object]] = []
    predictions: List[pd.DataFrame] = []

    for source, articles in sentiment_sources.items():
        for view in LANGUAGE_VIEWS:
            daily = build_market_daily_features(articles, constituents, view)
            daily.to_csv(output_dir / "features" / f"{source}_{view}_equal_weight_daily.csv", index=False)
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
                artifact = f"h{horizon}_{source}_{view}"
                panel, feature_columns = build_enhanced_sentiment_panel(prices, daily, config)
                banned = sorted(column for column in feature_columns if any(token in column.lower() for token in BANNED_FEATURE_TOKENS))
                # Sentiment volume is article volume, not market trading volume.
                banned = [column for column in banned if "sentiment_weighted_volume" not in column]
                if banned:
                    raise RuntimeError(f"Non-sentiment feature columns entered the model matrix: {banned}")
                panel.to_csv(output_dir / "panels" / f"{artifact}_panel.csv", index=False)
                split = build_time_series_split(len(panel), config.split_config, config.horizon_days)
                selected_c, validation_direction = _select_best_direction_hyperparameter(panel, feature_columns, split, config)
                selected_alpha, validation_return = _select_best_return_hyperparameter(panel, feature_columns, split, config)
                threshold = choose_validation_threshold(validation_direction, threshold_grid)
                direction = apply_classification_threshold(
                    pd.concat(
                        [
                            fit_in_sample_direction_predictions(panel, feature_columns, split.train_end, selected_c, config.random_seed),
                            validation_direction,
                            walk_forward_direction_predictions(
                                panel, feature_columns, split.test_indices, config.horizon_days, selected_c,
                                config.random_seed, "test", config.refit_every_rows,
                            ),
                        ],
                        ignore_index=True,
                    ),
                    threshold,
                )
                returns = pd.concat(
                    [
                        fit_in_sample_return_predictions(panel, feature_columns, split.train_end, selected_alpha),
                        validation_return,
                        walk_forward_return_predictions(
                            panel, feature_columns, split.test_indices, config.horizon_days, selected_alpha,
                            "test", config.refit_every_rows,
                        ),
                    ],
                    ignore_index=True,
                )
                for split_label in ("validation", "test"):
                    metric_rows.append(
                        {
                            "horizon": int(horizon),
                            "sentiment_source": source,
                            "view": view,
                            "split": split_label,
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                            "selected_C": float(selected_c),
                            "classification_threshold": float(threshold),
                            **compute_direction_metrics(direction[direction["split"] == split_label]),
                        }
                    )
                    return_rows.append(
                        {
                            "horizon": int(horizon),
                            "sentiment_source": source,
                            "view": view,
                            "split": split_label,
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                            "selected_alpha": float(selected_alpha),
                            **compute_return_metrics(returns[returns["split"] == split_label]),
                        }
                    )
                predictions.append(
                    _merge_predictions(
                        direction, returns, horizon=int(horizon), view=view, source=source,
                        selected_c=selected_c, selected_alpha=selected_alpha, threshold=threshold,
                    )
                )

    direction_metrics = pd.DataFrame(metric_rows)
    return_metrics = pd.DataFrame(return_rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    direction_metrics.to_csv(output_dir / "metrics" / "direction_metrics.csv", index=False)
    return_metrics.to_csv(output_dir / "metrics" / "return_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "predictions" / "predictions.csv", index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    direction_metrics.to_csv(metrics_path, index=False)
    summary = {
        "target": EGX30_TICKER,
        "target_usage": "labels_only",
        "constituent_validation": validation["status"],
        "sentiment_sources": list(sentiment_sources),
        "processed_articles": int(len(scored)),
        "pit_scoped_articles": {source: int(len(value)) for source, value in sentiment_sources.items()},
        "direction_metrics_path": str(metrics_path),
        "return_metrics_path": str(output_dir / "metrics" / "return_metrics.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if update_state:
        update_pipeline_state(
            current_phase="phase_4_strict_walk_forward_evaluation",
            status="completed",
            completed_phases=["phase_4_strict_walk_forward_evaluation"],
            artifacts={"universe_metrics": str(metrics_path), "universe_evaluation_summary": str(output_dir / "summary.json")},
        )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_experiments(
        processed_news_path=args.processed_news_path,
        constituent_path=args.constituent_path,
        target_price_path=args.target_price_path,
        llm_cache_path=args.llm_cache_path,
        camelbert_processed_news_path=args.camelbert_processed_news_path,
        output_dir=args.output_dir,
        metrics_path=args.metrics_path,
        horizons=args.horizons,
        rolling_windows=args.rolling_windows,
        classification_c_grid=args.classification_c_grid,
        regression_alpha_grid=args.regression_alpha_grid,
        threshold_grid=args.threshold_grid,
        refit_every_rows=args.refit_every_rows,
        update_state=not args.skip_state_update,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
