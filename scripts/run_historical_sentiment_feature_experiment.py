"""Compare historical-only EGX direction models against sentiment-augmented variants.

This experiment documents the earlier Robin/Beltone-style integration test:
use the historical feature pipeline from `anastamerr/EGX-Direction-Forecasting`
and add the thesis sentiment pipeline as an additional lagged signal.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


PRICE_FILES = {
    "CIB": "Commercial Int Bank Stock Price History",
    "TMG": "TMG Holding Stock Price History",
    "OIH": "Orascom Investment Holding",
}

SENTIMENT_BASE_COLUMNS = (
    "combined_article_count",
    "combined_scored_count",
    "combined_sentiment_weighted_mean",
    "combined_sentiment_weighted_volume",
    "combined_sentiment_polarity_share",
    "combined_confidence_mean",
    "combined_relevance_weight_mean",
    "en_sentiment_weighted_mean",
    "ar_sentiment_weighted_mean",
    "en_ar_sentiment_spread",
    "en_share_of_articles",
    "ar_share_of_articles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historical baseline vs sentiment-augmented EGX direction experiments."
    )
    parser.add_argument(
        "--direction-repo-path",
        default="/tmp/EGX-Direction-Forecasting-experiment",
        help="Local checkout of https://github.com/anastamerr/EGX-Direction-Forecasting.",
    )
    parser.add_argument("--data-dir", default="data", help="Thesis price data directory.")
    parser.add_argument(
        "--daily-sentiment-path",
        default="outputs/final_sentiment_experiments/features/h5_all_news_bilingual_daily_features.csv",
        help="Daily bilingual sentiment features produced by the thesis pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/historical_sentiment_feature_experiment",
        help="Directory for experiment artifacts.",
    )
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--models", nargs="+", default=["logistic", "xgboost", "mlp"], choices=["logistic", "xgboost", "mlp"])
    parser.add_argument("--nn-epochs", type=int, default=50)
    parser.add_argument("--past-window", type=int, default=10)
    parser.add_argument("--future-window", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--gap", type=int, default=None)
    return parser.parse_args()


def import_direction_repo(direction_repo_path: Path):
    if not direction_repo_path.exists():
        raise FileNotFoundError(
            f"Direction repo checkout not found at {direction_repo_path}. "
            "Clone https://github.com/anastamerr/EGX-Direction-Forecasting first."
        )
    repo_str = str(direction_repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from preprocessing import BasePreprocessor  # type: ignore
    from models.logistic_regression import LogisticRegressionModel  # type: ignore
    from models.xgboost_model import XGBoostModel  # type: ignore
    from models.neural_network import NeuralNetworkModel  # type: ignore

    return BasePreprocessor, LogisticRegressionModel, XGBoostModel, NeuralNetworkModel


def load_price_frames(data_dir: Path, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        matches = sorted(data_dir.glob(f"{PRICE_FILES[ticker]}*.csv"))
        if not matches:
            raise FileNotFoundError(f"No local price CSV found for {ticker} using stem {PRICE_FILES[ticker]!r}")
        frame = pd.read_csv(matches[0])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame["Close"] = pd.to_numeric(frame["Close"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if "Volume" in frame.columns:
            volume = frame["Volume"].astype(str).str.replace(",", "", regex=False).str.strip()
            volume = volume.str.replace("K", "e3", regex=False).str.replace("M", "e6", regex=False)
            frame["Volume"] = pd.to_numeric(volume, errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        frame["Ticker"] = ticker
        frames[ticker] = frame
    return frames


def _days_since_signal(series: pd.Series) -> pd.Series:
    values: List[float] = []
    seen: Optional[int] = None
    default = max(len(series), 1)
    for flag in series.fillna(0.0).astype(float).gt(0.0).tolist():
        if flag:
            seen = 0
        elif seen is None:
            seen = default
        else:
            seen += 1
        values.append(float(seen))
    return pd.Series(values, index=series.index).clip(upper=365.0)


def build_lagged_sentiment_features(
    price_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    rolling_windows: Sequence[int] = (3, 5, 7),
) -> Tuple[pd.DataFrame, List[str]]:
    """Align sentiment to trading dates with a one-calendar-day lag."""

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    ticker = str(prices["Ticker"].iloc[0])
    price_dates = pd.Index(prices["Date"])

    daily = daily_sentiment[daily_sentiment["Ticker"] == ticker].copy()
    base_columns = [col for col in SENTIMENT_BASE_COLUMNS if col in daily_sentiment.columns]
    if daily.empty:
        calendar_index = pd.date_range(price_dates.min(), price_dates.max(), freq="D")
        base_daily = pd.DataFrame(0.0, index=calendar_index, columns=base_columns)
    else:
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
        calendar_index = pd.date_range(min(price_dates.min(), daily["Date"].min()), price_dates.max(), freq="D")
        base_daily = daily.set_index("Date")[base_columns].reindex(calendar_index).fillna(0.0)

    shifted = base_daily.shift(1).fillna(0.0)
    feature_data: Dict[str, pd.Series] = {}
    for column in base_columns:
        feature_data[f"sent_{column}_lag1d"] = shifted[column]
        for window in rolling_windows:
            if column.endswith("_count") or column.endswith("_volume"):
                feature_data[f"sent_{column}_roll{window}d"] = (
                    shifted[column].rolling(window, min_periods=1).sum().fillna(0.0)
                )
            else:
                feature_data[f"sent_{column}_roll{window}d"] = (
                    shifted[column].rolling(window, min_periods=1).mean().fillna(0.0)
                )

    if "combined_article_count" in shifted.columns:
        feature_data["sent_days_since_article"] = _days_since_signal(shifted["combined_article_count"])

    aligned = pd.DataFrame(feature_data, index=base_daily.index)
    aligned = aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=-1000.0, upper=1000.0)
    aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": "Date"})
    aligned["Ticker"] = ticker
    sentiment_columns = [column for column in aligned.columns if column not in {"Date", "Ticker"}]
    return aligned[["Date", "Ticker"] + sentiment_columns], sentiment_columns


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_proba),
    }


def make_model(model_name: str, LogisticRegressionModel, XGBoostModel, NeuralNetworkModel, input_dim: int, nn_epochs: int):
    if model_name == "logistic":
        return LogisticRegressionModel(C=1.0, max_iter=1000, class_weight="balanced", scale=False)
    if model_name == "xgboost":
        return XGBoostModel(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            auto_scale_pos_weight=True,
            random_state=42,
        )
    if model_name == "mlp":
        np.random.seed(42)
        try:
            import torch

            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        return NeuralNetworkModel(
            architecture="mlp",
            input_dim=input_dim,
            epochs=nn_epochs,
            batch_size=128,
            learning_rate=1e-3,
            validation_fraction=0.1,
            patience=8,
            class_weight="balanced",
            random_state=42,
            device="cpu",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def feature_importance_rows(model_name: str, model, feature_columns: Sequence[str], limit: int = 25) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if model_name == "logistic":
        fitted = model.model
        coef = getattr(fitted, "coef_", None)
        if coef is None and hasattr(fitted, "named_steps"):
            coef = fitted.named_steps["clf"].coef_
        if coef is not None:
            for feature, value in sorted(zip(feature_columns, coef[0]), key=lambda item: abs(item[1]), reverse=True)[:limit]:
                rows.append({"feature": feature, "importance": float(value), "abs_importance": float(abs(value))})
    elif model_name == "xgboost":
        booster = getattr(model.model, "feature_importances_", None)
        if booster is not None:
            for feature, value in sorted(zip(feature_columns, booster), key=lambda item: item[1], reverse=True)[:limit]:
                rows.append({"feature": feature, "importance": float(value), "abs_importance": float(value)})
    return rows


def run_one_variant(
    *,
    ticker: str,
    price_df: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    sentiment_columns: Sequence[str],
    variant: str,
    model_name: str,
    BasePreprocessor,
    LogisticRegressionModel,
    XGBoostModel,
    NeuralNetworkModel,
    past_window: int,
    future_window: int,
    test_size: float,
    gap: Optional[int],
    nn_epochs: int,
) -> Tuple[Dict[str, object], pd.DataFrame, List[Dict[str, object]]]:
    pre = BasePreprocessor(past_window=past_window, future_window=future_window)
    processed, historical_features = pre.create_features_and_target(price_df)
    processed["Date"] = pd.to_datetime(processed["Date"], errors="coerce").dt.normalize()

    if variant == "historical_sentiment":
        panel = processed.merge(sentiment_features, on=["Date", "Ticker"], how="left")
        for column in sentiment_columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
        features = list(historical_features) + list(sentiment_columns)
    elif variant == "historical_only":
        panel = processed.copy()
        features = list(historical_features)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    panel[features] = panel[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gap_used = future_window + max(pre.return_lags) if gap is None else gap
    split = pre.train_test_split(panel, features, test_size=test_size, gap=gap_used)

    model = make_model(model_name, LogisticRegressionModel, XGBoostModel, NeuralNetworkModel, len(features), nn_epochs)
    model.train(split["X_train"].values, split["y_train"].values)
    y_pred = model.predict(split["X_test"].values)
    y_proba = model.predict_proba(split["X_test"].values)[:, 1]
    y_true = split["y_test"].values

    metrics = compute_metrics(y_true, y_pred, y_proba)
    test_dates = pd.to_datetime(split["test_dates"], errors="coerce").dt.normalize()
    predictions = pd.DataFrame(
        {
            "ticker": ticker,
            "variant": variant,
            "model": model_name,
            "Date": test_dates.to_numpy(),
            "actual_direction": y_true,
            "predicted_direction": y_pred,
            "predicted_probability": y_proba,
        }
    )
    summary = {
        "ticker": ticker,
        "variant": variant,
        "model": model_name,
        "past_window": past_window,
        "future_window": future_window,
        "gap": gap_used,
        "train_rows": int(len(split["X_train"])),
        "test_rows": int(len(split["X_test"])),
        "feature_count": int(len(features)),
        "historical_feature_count": int(len(historical_features)),
        "sentiment_feature_count": int(len(sentiment_columns) if variant == "historical_sentiment" else 0),
        "test_start": test_dates.min().date().isoformat() if len(test_dates) else "",
        "test_end": test_dates.max().date().isoformat() if len(test_dates) else "",
        **metrics,
    }
    importance = feature_importance_rows(model_name, model, features)
    for row in importance:
        row.update({"ticker": ticker, "variant": variant, "model": model_name})
    return summary, predictions, importance


def build_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    key = ["ticker", "model"]
    baseline = metrics[metrics["variant"] == "historical_only"].copy()
    augmented = metrics[metrics["variant"] == "historical_sentiment"].copy()
    merged = baseline.merge(augmented, on=key, suffixes=("_historical", "_sentiment"))
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"):
        merged[f"delta_{metric}"] = merged[f"{metric}_sentiment"] - merged[f"{metric}_historical"]
    return merged.sort_values(["model", "ticker"]).reset_index(drop=True)


def write_markdown(metrics: pd.DataFrame, deltas: pd.DataFrame, output_path: Path) -> None:
    aggregate = deltas.groupby("model")[[f"delta_{m}" for m in ("accuracy", "balanced_accuracy", "f1", "auc")]].mean()
    best_accuracy = deltas.sort_values("delta_accuracy", ascending=False).iloc[0]
    worst_accuracy = deltas.sort_values("delta_accuracy", ascending=True).iloc[0]
    improved_accuracy_rows = int((deltas["delta_accuracy"] > 0).sum())
    total_delta_rows = int(len(deltas))

    lines = [
        "# Historical Model Sentiment Feature Experiment",
        "",
        "This experiment re-runs the historical EGX direction model from `anastamerr/EGX-Direction-Forecasting` and compares a historical-only feature set against the same feature set augmented with one-day-lagged bilingual sentiment features from the thesis pipeline.",
        "",
        "The target and chronological train/test split are shared between both variants for each ticker/model pair.",
        "",
        "## Test Metrics",
        "",
        "| Ticker | Model | Variant | Features | Accuracy | Balanced Acc. | F1 | AUC | Test period |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in metrics.sort_values(["model", "ticker", "variant"]).iterrows():
        lines.append(
            f"| {row['ticker']} | {row['model']} | {row['variant']} | {int(row['feature_count'])} | "
            f"{row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['f1']:.3f} | {row['auc']:.3f} | "
            f"{row['test_start']} to {row['test_end']} |"
        )

    lines.extend(
        [
            "",
            "## Sentiment Delta",
            "",
            "| Ticker | Model | Delta Accuracy | Delta Balanced Acc. | Delta F1 | Delta AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in deltas.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['model']} | {row['delta_accuracy']:+.3f} | "
            f"{row['delta_balanced_accuracy']:+.3f} | {row['delta_f1']:+.3f} | {row['delta_auc']:+.3f} |"
        )

    lines.extend(["", "## Aggregate Mean Delta", "", "| Model | Accuracy | Balanced Acc. | F1 | AUC |", "| --- | ---: | ---: | ---: | ---: |"])
    for model, row in aggregate.iterrows():
        lines.append(
            f"| {model} | {row['delta_accuracy']:+.3f} | {row['delta_balanced_accuracy']:+.3f} | "
            f"{row['delta_f1']:+.3f} | {row['delta_auc']:+.3f} |"
        )

    auc_delta_mean = float(deltas["delta_auc"].mean()) if "delta_auc" in deltas else float("nan")
    auc_sentence = (
        "Mean AUC improves slightly overall."
        if auc_delta_mean > 0
        else "Mean AUC declines overall."
        if auc_delta_mean < 0
        else "Mean AUC is unchanged overall."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Sentiment improved raw test accuracy in {improved_accuracy_rows} of {total_delta_rows} ticker/model comparisons. "
            f"The largest accuracy gain was `{best_accuracy['delta_accuracy']:+.3f}` for `{best_accuracy['ticker']}` using `{best_accuracy['model']}`. "
            f"The largest accuracy loss was `{worst_accuracy['delta_accuracy']:+.3f}` for `{worst_accuracy['ticker']}` using `{worst_accuracy['model']}`.",
            "",
            f"The result should be described as conditional rather than uniformly positive. The sentiment signal adds small gains for several model/ticker pairs, but it does not consistently improve every ticker or every metric. {auc_sentence}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_metadata(repo_path: Path) -> Dict[str, str]:
    def run_git(args: Sequence[str]) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(repo_path), *args], text=True).strip()
        except Exception:
            return ""

    return {
        "repo_path": str(repo_path),
        "remote_origin": run_git(["remote", "get-url", "origin"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    BasePreprocessor, LogisticRegressionModel, XGBoostModel, NeuralNetworkModel = import_direction_repo(Path(args.direction_repo_path))
    daily_sentiment = pd.read_csv(args.daily_sentiment_path)
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"], errors="coerce").dt.normalize()
    price_frames = load_price_frames(Path(args.data_dir), args.tickers)

    metric_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    importance_rows: List[Dict[str, object]] = []

    for ticker in args.tickers:
        sentiment_features, sentiment_columns = build_lagged_sentiment_features(price_frames[ticker], daily_sentiment)
        for model_name in args.models:
            for variant in ("historical_only", "historical_sentiment"):
                print(f"Running ticker={ticker} model={model_name} variant={variant}", flush=True)
                metrics, predictions, importance = run_one_variant(
                    ticker=ticker,
                    price_df=price_frames[ticker],
                    sentiment_features=sentiment_features,
                    sentiment_columns=sentiment_columns,
                    variant=variant,
                    model_name=model_name,
                    BasePreprocessor=BasePreprocessor,
                    LogisticRegressionModel=LogisticRegressionModel,
                    XGBoostModel=XGBoostModel,
                    NeuralNetworkModel=NeuralNetworkModel,
                    past_window=args.past_window,
                    future_window=args.future_window,
                    test_size=args.test_size,
                    gap=args.gap,
                    nn_epochs=args.nn_epochs,
                )
                metric_rows.append(metrics)
                prediction_frames.append(predictions)
                importance_rows.extend(importance)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importance = pd.DataFrame(importance_rows)
    deltas = build_delta_table(metrics)

    metrics_path = output_dir / "historical_sentiment_metrics.csv"
    predictions_path = output_dir / "historical_sentiment_predictions.csv"
    deltas_path = output_dir / "historical_sentiment_deltas.csv"
    importance_path = output_dir / "historical_sentiment_feature_importance.csv"
    summary_path = output_dir / "historical_sentiment_summary.md"
    config_path = output_dir / "run_config.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    importance.to_csv(importance_path, index=False)
    write_markdown(metrics, deltas, summary_path)
    run_config = vars(args).copy()
    run_config["direction_repo_metadata"] = git_metadata(Path(args.direction_repo_path))
    run_config["sentiment_base_columns"] = list(SENTIMENT_BASE_COLUMNS)
    config_path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved deltas: {deltas_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
