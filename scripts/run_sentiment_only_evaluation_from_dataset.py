"""Run sentiment-only forecasting evaluation from an existing labeled article dataset."""

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

from data_loader import StockDataLoader
from egx_direction.sentiment_only_pipeline import (
    LANGUAGE_VIEWS,
    SentimentOnlyPipelineConfig,
    aggregate_daily_sentiment_features,
    build_sentiment_targets,
    build_sentiment_panel,
    build_time_series_split,
    compute_direction_metrics,
    compute_return_metrics,
    fit_in_sample_direction_predictions,
    fit_in_sample_return_predictions,
    walk_forward_direction_predictions,
    walk_forward_return_predictions,
    _extract_feature_matrix,
    _fit_direction_model,
    _fit_return_model,
    _select_best_direction_hyperparameter,
    _select_best_return_hyperparameter,
)
from news_pipeline import resolve_ticker


ENHANCED_PREFIXES = ("combined", "en", "ar")
ENHANCED_ROLLING_WINDOWS = (3, 5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EGX sentiment-only forecasting from a cleaned sentiment-labeled dataset."
    )
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
        help="Cleaned sentiment-labeled article CSV produced by the sentiment-ready pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sentiment_only_dataset_evaluation",
        help="Output directory for no-scrape evaluation artifacts.",
    )
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"], help="Target EGX tickers.")
    parser.add_argument("--price-period", default="10y", help="yfinance period if local cached prices are unavailable.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days.")
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=list(ENHANCED_ROLLING_WINDOWS),
        help="Rolling calendar windows used for sentiment features.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["baseline", "enhanced"],
        default="enhanced",
        help="Feature engineering mode. Baseline keeps the original feature set; enhanced adds lag/rolling/volume/momentum features.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="outputs/sentiment_only_dataset_evaluation",
        help="Existing baseline output directory used for old-vs-new comparison.",
    )
    parser.add_argument(
        "--classification-c-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.5, 1.0],
        help="Small validation grid for logistic regression C.",
    )
    parser.add_argument(
        "--regression-alpha-grid",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0],
        help="Small validation grid for ridge regression alpha.",
    )
    return parser.parse_args()


def load_scored_articles(path: Path, config: SentimentOnlyPipelineConfig) -> pd.DataFrame:
    """Map the public sentiment-ready dataset into the internal article schema."""

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No rows found in sentiment-labeled dataset: {path}")

    scored = pd.DataFrame()
    scored["Ticker"] = frame.get("canonical_ticker", frame.get("ticker")).map(resolve_ticker)
    scored = scored[scored["Ticker"].isin(config.tickers)].copy()
    source = frame.loc[scored.index].copy()

    published_at = pd.to_datetime(source["published_timestamp"], utc=True, errors="coerce")
    scored["published_at"] = published_at
    scored["Date"] = published_at.dt.tz_convert(config.timezone).dt.tz_localize(None).dt.normalize()
    scored["article_id"] = source["article_id"].astype(str)
    scored["article_url"] = source.get("article_url", source.get("url", "")).fillna("")
    for text_column in (
        "title",
        "body_text",
        "cleaned_title",
        "cleaned_body_text",
        "clean_article_text",
    ):
        scored[text_column] = source.get(text_column, "").fillna("").astype(str)
    scored["article_language"] = source.get("article_language", source.get("language", "")).fillna("").astype(str)
    scored["detected_language"] = source["detected_language"].fillna("").astype(str)
    scored["sentiment_label"] = source["sentiment_label"].fillna("unscored").astype(str)
    scored["positive_prob"] = pd.to_numeric(source["positive_probability"], errors="coerce").fillna(0.0)
    scored["negative_prob"] = pd.to_numeric(source["negative_probability"], errors="coerce").fillna(0.0)
    scored["neutral_prob"] = pd.to_numeric(source["neutral_probability"], errors="coerce").fillna(0.0)
    scored["sentiment_score"] = pd.to_numeric(source["continuous_sentiment_score"], errors="coerce").fillna(0.0)
    scored["sentiment_score_standardized"] = scored["sentiment_score"]
    scored["confidence"] = pd.to_numeric(source.get("confidence", 0.0), errors="coerce").fillna(0.0)
    scored["is_scored"] = pd.to_numeric(source.get("is_scored", 0), errors="coerce").fillna(0).astype(int)
    scored["entity_relevance_label"] = source.get("entity_relevance_label", "indirect").fillna("indirect").astype(str)
    scored["entity_relevance_weight"] = pd.to_numeric(source.get("relevance_score", 0.0), errors="coerce").fillna(0.0)
    scored["sentiment_article_weight"] = (
        scored["entity_relevance_weight"] * scored["confidence"] * scored["is_scored"]
    )

    scored = scored.dropna(subset=["Ticker", "Date", "published_at"])
    scored = scored.sort_values(["Ticker", "Date", "published_at", "article_id"]).reset_index(drop=True)
    return scored


def load_price_frames(config: SentimentOnlyPipelineConfig, output_dir: Path) -> Dict[str, pd.DataFrame]:
    local_loader = StockDataLoader(data_dir="data")
    fresh_loader = StockDataLoader(data_dir=str(output_dir / "prices"))
    price_dir = output_dir / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    for ticker in config.tickers:
        try:
            price_df = local_loader.load_ticker(
                ticker,
                allow_yfinance_fallback=False,
                period=config.price_period,
            ).copy()
        except Exception:
            price_df = fresh_loader.load_ticker(
                ticker,
                allow_yfinance_fallback=True,
                period=config.price_period,
            ).copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.normalize()
        price_df = price_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        price_df["Ticker"] = ticker
        price_df.to_csv(price_dir / f"{ticker}_price_history.csv", index=False)
        frames[ticker] = price_df
    return frames


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_values = values.fillna(0.0).astype(float).to_numpy()
    clean_weights = weights.fillna(0.0).astype(float).to_numpy()
    if len(clean_values) == 0:
        return 0.0
    if clean_weights.sum() <= 0:
        return float(clean_values.mean())
    return float(np.average(clean_values, weights=clean_weights))


def _empty_article_aggregate(prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_article_count": 0.0,
        f"{prefix}_scored_count": 0.0,
        f"{prefix}_positive_count": 0.0,
        f"{prefix}_negative_count": 0.0,
        f"{prefix}_neutral_count": 0.0,
        f"{prefix}_sentiment_mean": 0.0,
        f"{prefix}_sentiment_weighted_mean": 0.0,
        f"{prefix}_sentiment_sum": 0.0,
        f"{prefix}_sentiment_abs_mean": 0.0,
        f"{prefix}_sentiment_std": 0.0,
        f"{prefix}_positive_prob_mean": 0.0,
        f"{prefix}_negative_prob_mean": 0.0,
        f"{prefix}_neutral_prob_mean": 0.0,
        f"{prefix}_confidence_mean": 0.0,
        f"{prefix}_relevance_weight_total": 0.0,
        f"{prefix}_relevance_weight_mean": 0.0,
        f"{prefix}_sentiment_weighted_volume": 0.0,
        f"{prefix}_sentiment_polarity_count": 0.0,
        f"{prefix}_sentiment_polarity_share": 0.0,
    }


def _aggregate_article_subset(group: pd.DataFrame, prefix: str) -> Dict[str, float]:
    if group.empty:
        return _empty_article_aggregate(prefix)

    scored = group[group["is_scored"] == 1].copy()
    row = _empty_article_aggregate(prefix)
    row[f"{prefix}_article_count"] = float(len(group))
    row[f"{prefix}_scored_count"] = float(len(scored))
    row[f"{prefix}_relevance_weight_total"] = float(group["entity_relevance_weight"].fillna(0.0).sum())
    row[f"{prefix}_relevance_weight_mean"] = float(group["entity_relevance_weight"].fillna(0.0).mean())

    if scored.empty:
        return row

    scores = scored["sentiment_score_standardized"].fillna(0.0).astype(float)
    weights = scored["sentiment_article_weight"].fillna(0.0).astype(float)
    positive_count = float((scored["sentiment_label"] == "positive").sum())
    negative_count = float((scored["sentiment_label"] == "negative").sum())
    neutral_count = float((scored["sentiment_label"] == "neutral").sum())

    row.update(
        {
            f"{prefix}_positive_count": positive_count,
            f"{prefix}_negative_count": negative_count,
            f"{prefix}_neutral_count": neutral_count,
            f"{prefix}_sentiment_mean": float(scores.mean()),
            f"{prefix}_sentiment_weighted_mean": _safe_weighted_mean(scores, weights),
            f"{prefix}_sentiment_sum": float(scores.sum()),
            f"{prefix}_sentiment_abs_mean": float(scores.abs().mean()),
            f"{prefix}_sentiment_std": float(scores.std(ddof=0)),
            f"{prefix}_positive_prob_mean": float(scored["positive_prob"].fillna(0.0).astype(float).mean()),
            f"{prefix}_negative_prob_mean": float(scored["negative_prob"].fillna(0.0).astype(float).mean()),
            f"{prefix}_neutral_prob_mean": float(scored["neutral_prob"].fillna(0.0).astype(float).mean()),
            f"{prefix}_confidence_mean": float(scored["confidence"].fillna(0.0).astype(float).mean()),
            f"{prefix}_sentiment_weighted_volume": float((scores * weights).sum()),
            f"{prefix}_sentiment_polarity_count": positive_count - negative_count,
            f"{prefix}_sentiment_polarity_share": (positive_count - negative_count) / max(len(scored), 1),
        }
    )
    return row


def aggregate_enhanced_daily_sentiment_features(scored_articles: pd.DataFrame, view: str) -> pd.DataFrame:
    """Aggregate daily sentiment with separate combined, English, and Arabic signals."""

    if view == "english_only":
        subset = scored_articles[scored_articles["detected_language"] == "en"].copy()
    elif view == "arabic_only":
        subset = scored_articles[scored_articles["detected_language"] == "ar"].copy()
    elif view == "bilingual":
        subset = scored_articles[scored_articles["detected_language"].isin({"en", "ar"})].copy()
    else:
        raise ValueError(f"Unsupported view: {view}")

    if subset.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])

    rows: List[Dict[str, float]] = []
    for (ticker, date), group in subset.groupby(["Ticker", "Date"], sort=True):
        en_group = group[group["detected_language"] == "en"]
        ar_group = group[group["detected_language"] == "ar"]
        row: Dict[str, float] = {"Ticker": ticker, "Date": date}
        row.update(_aggregate_article_subset(group, "combined"))
        row.update(_aggregate_article_subset(en_group, "en"))
        row.update(_aggregate_article_subset(ar_group, "ar"))
        row["en_ar_sentiment_spread"] = row["en_sentiment_weighted_mean"] - row["ar_sentiment_weighted_mean"]
        row["en_ar_article_count_spread"] = row["en_article_count"] - row["ar_article_count"]
        row["en_share_of_articles"] = row["en_article_count"] / max(row["combined_article_count"], 1.0)
        row["ar_share_of_articles"] = row["ar_article_count"] / max(row["combined_article_count"], 1.0)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _enhanced_base_columns() -> List[str]:
    columns: List[str] = []
    for prefix in ENHANCED_PREFIXES:
        columns.extend(_empty_article_aggregate(prefix).keys())
    columns.extend(
        [
            "en_ar_sentiment_spread",
            "en_ar_article_count_spread",
            "en_share_of_articles",
            "ar_share_of_articles",
        ]
    )
    return columns


def align_enhanced_sentiment_features_to_prices(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    """Create leakage-safe lag/rolling/momentum sentiment features on trading dates."""

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    ticker = str(prices["Ticker"].iloc[0])
    price_dates = pd.Index(prices["Date"])
    base_columns = _enhanced_base_columns()

    if daily_features.empty:
        start_date = price_dates.min()
        base_daily = pd.DataFrame(index=pd.date_range(start=start_date, end=price_dates.max(), freq="D"))
        for column in base_columns:
            base_daily[column] = 0.0
    else:
        daily = daily_features.copy()
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
        start_date = min(price_dates.min(), daily["Date"].min())
        calendar_index = pd.date_range(start=start_date, end=price_dates.max(), freq="D")
        for column in base_columns:
            if column not in daily.columns:
                daily[column] = 0.0
        base_daily = daily.set_index("Date")[base_columns].reindex(calendar_index).fillna(0.0)

    feature_data: Dict[str, pd.Series] = {}

    for lag in (1, 2, 3):
        lagged = base_daily.shift(lag).fillna(0.0)
        for column in base_columns:
            feature_data[f"{column}_lag{lag}d"] = lagged[column]
        for prefix in ENHANCED_PREFIXES:
            feature_data[f"{prefix}_sentiment_t_minus_{lag}"] = lagged[f"{prefix}_sentiment_weighted_mean"]

    shifted = base_daily.shift(1).fillna(0.0)
    windows = tuple(sorted(set(int(window) for window in rolling_windows)))
    for prefix in ENHANCED_PREFIXES:
        sentiment = shifted[f"{prefix}_sentiment_weighted_mean"]
        article_count = shifted[f"{prefix}_article_count"]
        weighted_volume = shifted[f"{prefix}_sentiment_weighted_volume"]
        for window in windows:
            feature_data[f"{prefix}_sentiment_roll_mean_{window}d"] = sentiment.rolling(window, min_periods=1).mean().fillna(0.0)
            feature_data[f"{prefix}_sentiment_roll_std_{window}d"] = sentiment.rolling(window, min_periods=2).std(ddof=0).fillna(0.0)
            feature_data[f"{prefix}_article_count_roll_sum_{window}d"] = article_count.rolling(window, min_periods=1).sum().fillna(0.0)
            feature_data[f"{prefix}_article_count_roll_mean_{window}d"] = article_count.rolling(window, min_periods=1).mean().fillna(0.0)
            feature_data[f"{prefix}_sentiment_weighted_volume_roll_sum_{window}d"] = weighted_volume.rolling(window, min_periods=1).sum().fillna(0.0)

        previous = base_daily[f"{prefix}_sentiment_weighted_mean"].shift(1).fillna(0.0)
        prior = base_daily[f"{prefix}_sentiment_weighted_mean"].shift(2).fillna(0.0)
        earlier = base_daily[f"{prefix}_sentiment_weighted_mean"].shift(3).fillna(0.0)
        delta = previous - prior
        feature_data[f"{prefix}_sentiment_delta"] = delta
        feature_data[f"{prefix}_sentiment_acceleration"] = delta - (prior - earlier)
        feature_data[f"{prefix}_sentiment_trend_direction"] = np.sign(delta)
        if 3 in windows and 7 in windows:
            short_mean = sentiment.rolling(3, min_periods=1).mean().fillna(0.0)
            long_mean = sentiment.rolling(7, min_periods=1).mean().fillna(0.0)
            feature_data[f"{prefix}_sentiment_roll3_minus_roll7"] = short_mean - long_mean

    feature_data["en_ar_sentiment_spread_lag1d"] = shifted["en_ar_sentiment_spread"]
    feature_data["en_ar_sentiment_spread_abs_lag1d"] = shifted["en_ar_sentiment_spread"].abs()
    feature_data["en_ar_article_count_spread_lag1d"] = shifted["en_ar_article_count_spread"]
    feature_data["en_share_of_articles_lag1d"] = shifted["en_share_of_articles"]
    feature_data["ar_share_of_articles_lag1d"] = shifted["ar_share_of_articles"]

    if "combined_article_count" in shifted.columns:
        has_article = shifted["combined_article_count"] > 0
        days_since = []
        seen = None
        default = max(len(shifted), 1)
        for flag in has_article.tolist():
            if flag:
                seen = 0
            elif seen is None:
                seen = default
            else:
                seen += 1
            days_since.append(float(seen))
        feature_data["days_since_combined_article"] = pd.Series(days_since, index=shifted.index).clip(upper=365.0)

    aligned = pd.DataFrame(feature_data, index=base_daily.index)
    aligned = aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=-1000.0, upper=1000.0)
    aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": "Date"})
    aligned["Ticker"] = ticker
    return aligned[["Date", "Ticker"] + [column for column in aligned.columns if column not in {"Date", "Ticker"}]]


def build_enhanced_sentiment_panel(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    config: SentimentOnlyPipelineConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    aligned_features = align_enhanced_sentiment_features_to_prices(
        price_df=price_df,
        daily_features=daily_features,
        rolling_windows=config.rolling_windows,
    )
    targets = build_sentiment_targets(price_df, horizon_days=config.horizon_days)
    panel = targets.merge(aligned_features, how="left", on=["Date", "Ticker"])
    excluded = {"Date", "Ticker", "Close", "target_return", "target_direction", "target_horizon_end_date"}
    feature_columns = [column for column in panel.columns if column not in excluded]
    panel[feature_columns] = panel[feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(float)
    panel[feature_columns] = panel[feature_columns].clip(lower=-1000.0, upper=1000.0)
    panel["target_direction"] = panel["target_direction"].astype(int)
    return panel.sort_values("Date").reset_index(drop=True), feature_columns


def metric_rows_for_ticker(
    ticker: str,
    view: str,
    direction_predictions: pd.DataFrame,
    return_predictions: pd.DataFrame,
    feature_count: int,
    panel_rows: int,
    selected_c: float,
    selected_alpha: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    direction_rows: List[Dict[str, object]] = []
    return_rows: List[Dict[str, object]] = []
    for split_label in ("validation", "test"):
        direction_metrics = compute_direction_metrics(direction_predictions[direction_predictions["split"] == split_label])
        return_metrics = compute_return_metrics(return_predictions[return_predictions["split"] == split_label])
        direction_rows.append(
            {
                "ticker": ticker,
                "view": view,
                "split": split_label,
                "feature_count": feature_count,
                "panel_rows": panel_rows,
                "selected_C": selected_c,
                **direction_metrics,
            }
        )
        return_rows.append(
            {
                "ticker": ticker,
                "view": view,
                "split": split_label,
                "feature_count": feature_count,
                "panel_rows": panel_rows,
                "selected_alpha": selected_alpha,
                **return_metrics,
            }
        )
    return direction_rows, return_rows


def add_pooled_metrics(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame, direction_predictions: pd.DataFrame, return_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pooled_direction_rows: List[Dict[str, object]] = []
    pooled_return_rows: List[Dict[str, object]] = []
    for view in LANGUAGE_VIEWS:
        for split_label in ("validation", "test"):
            direction_subset = direction_predictions[
                (direction_predictions["view"] == view) & (direction_predictions["split"] == split_label)
            ]
            return_subset = return_predictions[
                (return_predictions["view"] == view) & (return_predictions["split"] == split_label)
            ]
            pooled_direction_rows.append(
                {
                    "ticker": "ALL",
                    "view": view,
                    "split": split_label,
                    "feature_count": math.nan,
                    "panel_rows": math.nan,
                    "selected_C": math.nan,
                    **compute_direction_metrics(direction_subset),
                }
            )
            pooled_return_rows.append(
                {
                    "ticker": "ALL",
                    "view": view,
                    "split": split_label,
                    "feature_count": math.nan,
                    "panel_rows": math.nan,
                    "selected_alpha": math.nan,
                    **compute_return_metrics(return_subset),
                }
            )
    return (
        pd.concat([direction_metrics, pd.DataFrame(pooled_direction_rows)], ignore_index=True),
        pd.concat([return_metrics, pd.DataFrame(pooled_return_rows)], ignore_index=True),
    )


def build_ticker_comparison(direction_metrics: pd.DataFrame, return_metrics: pd.DataFrame) -> pd.DataFrame:
    direction_test = direction_metrics[
        (direction_metrics["ticker"] != "ALL")
        & (direction_metrics["view"] == "bilingual")
        & (direction_metrics["split"] == "test")
    ].copy()
    return_test = return_metrics[
        (return_metrics["ticker"] != "ALL")
        & (return_metrics["view"] == "bilingual")
        & (return_metrics["split"] == "test")
    ].copy()

    merged = direction_test.merge(
        return_test,
        on=["ticker", "view", "split"],
        suffixes=("_direction", "_return"),
    )
    comparison_columns = [
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
    ]
    return merged[comparison_columns].sort_values(
        ["balanced_accuracy", "f1", "sign_accuracy"],
        ascending=[False, False, False],
    )


def compute_feature_importance_rows(
    ticker: str,
    view: str,
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    split,
    selected_c: float,
    selected_alpha: float,
    random_seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    train_frame = panel.iloc[: split.train_end].dropna(subset=["target_direction", "target_return"]).copy()
    if train_frame.empty:
        return rows

    direction_model = _fit_direction_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=train_frame["target_direction"].astype(int).to_numpy(),
        regularization_c=selected_c,
        random_seed=random_seed,
    )
    if direction_model is not None:
        coefficients = direction_model.named_steps["classifier"].coef_[0]
        for feature, coefficient in zip(feature_columns, coefficients):
            rows.append(
                {
                    "ticker": ticker,
                    "view": view,
                    "task": "direction",
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )

    return_model = _fit_return_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=train_frame["target_return"].astype(float).to_numpy(),
        alpha=selected_alpha,
    )
    if return_model is not None:
        coefficients = return_model.named_steps["regressor"].coef_
        for feature, coefficient in zip(feature_columns, coefficients):
            rows.append(
                {
                    "ticker": ticker,
                    "view": view,
                    "task": "return",
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )
    return rows


def compare_against_baseline(output_dir: Path, baseline_dir: Path, new_comparison: pd.DataFrame) -> str:
    baseline_path = baseline_dir / "comparison" / "bilingual_test_ticker_comparison.csv"
    comparison_path = output_dir / "comparison" / "baseline_vs_enhanced_bilingual_test.csv"
    if not baseline_path.exists():
        new_comparison.assign(comparison_note=f"Baseline not found at {baseline_path}").to_csv(comparison_path, index=False)
        return str(comparison_path)

    baseline = pd.read_csv(baseline_path)
    merged = baseline.merge(new_comparison, on="ticker", suffixes=("_baseline", "_enhanced"))
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mae",
        "rmse",
        "sign_accuracy",
        "return_correlation",
    ):
        merged[f"delta_{metric}"] = merged[f"{metric}_enhanced"] - merged[f"{metric}_baseline"]
    merged.to_csv(comparison_path, index=False)
    return str(comparison_path)


def run_evaluation(
    config: SentimentOnlyPipelineConfig,
    sentiment_labeled_path: Path,
    *,
    feature_set: str,
    baseline_dir: Path,
) -> Dict[str, object]:
    output_dir = Path(config.output_dir)
    artifact_dirs = {
        "features": output_dir / "features",
        "panels": output_dir / "panels",
        "predictions": output_dir / "predictions",
        "metrics": output_dir / "metrics",
        "comparison": output_dir / "comparison",
    }
    for path in artifact_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    scored_articles = load_scored_articles(sentiment_labeled_path, config)
    scored_articles_path = output_dir / "articles_from_dataset.csv"
    scored_articles.to_csv(scored_articles_path, index=False)

    price_frames = load_price_frames(config, output_dir)
    direction_metric_rows: List[Dict[str, object]] = []
    return_metric_rows: List[Dict[str, object]] = []
    hyperparameter_rows: List[Dict[str, object]] = []
    direction_predictions_all: List[pd.DataFrame] = []
    return_predictions_all: List[pd.DataFrame] = []
    feature_importance_rows: List[Dict[str, object]] = []

    for view in LANGUAGE_VIEWS:
        if feature_set == "baseline":
            daily_features = aggregate_daily_sentiment_features(scored_articles=scored_articles, view=view)
        else:
            daily_features = aggregate_enhanced_daily_sentiment_features(scored_articles=scored_articles, view=view)
        daily_features.to_csv(artifact_dirs["features"] / f"daily_sentiment_features_{view}.csv", index=False)

        for ticker in config.tickers:
            ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
            if feature_set == "baseline":
                panel, feature_columns = build_sentiment_panel(
                    price_df=price_frames[ticker],
                    daily_features=ticker_daily,
                    config=config,
                )
            else:
                panel, feature_columns = build_enhanced_sentiment_panel(
                    price_df=price_frames[ticker],
                    daily_features=ticker_daily,
                    config=config,
                )
            panel.to_csv(artifact_dirs["panels"] / f"{ticker}_{view}_panel.csv", index=False)
            split = build_time_series_split(
                n_rows=len(panel),
                split_config=config.split_config,
                horizon_days=config.horizon_days,
            )

            selected_c, validation_direction = _select_best_direction_hyperparameter(panel, feature_columns, split, config)
            selected_alpha, validation_return = _select_best_return_hyperparameter(panel, feature_columns, split, config)
            hyperparameter_rows.extend(
                [
                    {
                        "ticker": ticker,
                        "view": view,
                        "task": "direction",
                        "selected_hyperparameter": selected_c,
                        "hyperparameter_name": "C",
                        "feature_count": len(feature_columns),
                        "panel_rows": len(panel),
                    },
                    {
                        "ticker": ticker,
                        "view": view,
                        "task": "return",
                        "selected_hyperparameter": selected_alpha,
                        "hyperparameter_name": "alpha",
                        "feature_count": len(feature_columns),
                        "panel_rows": len(panel),
                    },
                ]
            )

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
            direction_predictions = pd.concat([train_direction, validation_direction, test_direction], ignore_index=True)
            direction_predictions["view"] = view
            direction_predictions["ticker"] = ticker
            direction_predictions["task"] = "direction"
            direction_predictions_all.append(direction_predictions)

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
            return_predictions["view"] = view
            return_predictions["ticker"] = ticker
            return_predictions["task"] = "return"
            return_predictions_all.append(return_predictions)

            direction_rows, return_rows = metric_rows_for_ticker(
                ticker=ticker,
                view=view,
                direction_predictions=direction_predictions,
                return_predictions=return_predictions,
                feature_count=len(feature_columns),
                panel_rows=len(panel),
                selected_c=selected_c,
                selected_alpha=selected_alpha,
            )
            direction_metric_rows.extend(direction_rows)
            return_metric_rows.extend(return_rows)
            feature_importance_rows.extend(
                compute_feature_importance_rows(
                    ticker=ticker,
                    view=view,
                    panel=panel,
                    feature_columns=feature_columns,
                    split=split,
                    selected_c=selected_c,
                    selected_alpha=selected_alpha,
                    random_seed=config.random_seed,
                )
            )

    direction_predictions_df = pd.concat(direction_predictions_all, ignore_index=True)
    return_predictions_df = pd.concat(return_predictions_all, ignore_index=True)
    direction_predictions_path = artifact_dirs["predictions"] / "direction_predictions.csv"
    return_predictions_path = artifact_dirs["predictions"] / "return_predictions.csv"
    direction_predictions_df.to_csv(direction_predictions_path, index=False)
    return_predictions_df.to_csv(return_predictions_path, index=False)

    direction_metrics = pd.DataFrame(direction_metric_rows)
    return_metrics = pd.DataFrame(return_metric_rows)
    direction_metrics, return_metrics = add_pooled_metrics(
        direction_metrics,
        return_metrics,
        direction_predictions_df,
        return_predictions_df,
    )
    direction_metrics_path = artifact_dirs["metrics"] / "direction_metrics.csv"
    return_metrics_path = artifact_dirs["metrics"] / "return_metrics.csv"
    hyperparameter_path = artifact_dirs["metrics"] / "selected_hyperparameters.csv"
    direction_metrics.to_csv(direction_metrics_path, index=False)
    return_metrics.to_csv(return_metrics_path, index=False)
    pd.DataFrame(hyperparameter_rows).to_csv(hyperparameter_path, index=False)
    feature_importance_path = artifact_dirs["metrics"] / "feature_importance.csv"
    feature_importance = pd.DataFrame(feature_importance_rows)
    if not feature_importance.empty:
        feature_importance = feature_importance.sort_values(
            ["ticker", "view", "task", "abs_coefficient"],
            ascending=[True, True, True, False],
        )
    feature_importance.to_csv(feature_importance_path, index=False)

    comparison = build_ticker_comparison(direction_metrics, return_metrics)
    comparison_path = artifact_dirs["comparison"] / "bilingual_test_ticker_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    baseline_comparison_path = compare_against_baseline(output_dir, baseline_dir, comparison)

    summary = {
        "sentiment_labeled_path": str(sentiment_labeled_path),
        "output_dir": str(output_dir),
        "feature_set": feature_set,
        "tickers": list(config.tickers),
        "language_views": list(LANGUAGE_VIEWS),
        "horizon_days": config.horizon_days,
        "rolling_windows": list(config.rolling_windows),
        "scored_article_counts": scored_articles.groupby(["Ticker", "detected_language"]).size().reset_index(name="articles").to_dict(orient="records"),
        "artifacts": {
            "articles_from_dataset": str(scored_articles_path),
            "direction_metrics": str(direction_metrics_path),
            "return_metrics": str(return_metrics_path),
            "direction_predictions": str(direction_predictions_path),
            "return_predictions": str(return_predictions_path),
            "bilingual_test_ticker_comparison": str(comparison_path),
            "baseline_vs_enhanced_bilingual_test": baseline_comparison_path,
            "selected_hyperparameters": str(hyperparameter_path),
            "feature_importance": str(feature_importance_path),
        },
        "bilingual_test_ticker_comparison": comparison.to_dict(orient="records"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = SentimentOnlyPipelineConfig(
        tickers=tuple(args.tickers),
        output_dir=Path(args.output_dir),
        price_period=args.price_period,
        horizon_days=args.horizon,
        rolling_windows=tuple(args.rolling_windows),
        classification_c_grid=tuple(args.classification_c_grid),
        regression_alpha_grid=tuple(args.regression_alpha_grid),
    )
    summary = run_evaluation(
        config,
        Path(args.sentiment_labeled_path),
        feature_set=args.feature_set,
        baseline_dir=Path(args.baseline_dir),
    )
    print("Saved no-scrape evaluation to", summary["output_dir"])
    print("Feature set:", summary["feature_set"])
    print("Bilingual comparison:", summary["artifacts"]["bilingual_test_ticker_comparison"])
    print("Baseline comparison:", summary["artifacts"]["baseline_vs_enhanced_bilingual_test"])
    print("Feature importance:", summary["artifacts"]["feature_importance"])
    print("Direction metrics:", summary["artifacts"]["direction_metrics"])
    print("Return metrics:", summary["artifacts"]["return_metrics"])


if __name__ == "__main__":
    main()
