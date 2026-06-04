"""Additive easy-win audits for the EGX30 sentiment-only thesis pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from .market_index import EGX30_TICKER
from .sentiment_only_pipeline import (
    SentimentOnlyPipelineConfig,
    TimeSeriesSplit,
    build_time_series_split,
    compute_direction_metrics,
    walk_forward_direction_predictions,
)


REPORT_COLUMNS = [
    "horizon",
    "sentiment_source",
    "view",
    "split",
    "baseline_name",
    "rows",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "predicted_up_share",
    "actual_up_share",
    "majority_baseline_accuracy",
    "delta_vs_majority_baseline",
    "delta_balanced_accuracy_vs_training_majority",
    "delta_f1_vs_training_majority",
]


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    header = "| " + " | ".join(visible.columns) + " |"
    divider = "| " + " | ".join("---" for _ in visible.columns) + " |"
    rows = []
    for record in visible.to_dict("records"):
        values = []
        for column in visible.columns:
            value = record[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def load_frame(path: Path, date_columns: Sequence[str] = ("Date",)) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return frame


def company_coverage(scoped_articles: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    articles = scoped_articles.copy()
    articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce").dt.normalize()
    articles["Ticker"] = articles["Ticker"].astype(str).str.upper()
    company_names = (
        constituents.sort_values(["Ticker", "Date_From"])
        .drop_duplicates("Ticker")
        .set_index("Ticker")["Company_Name"]
        .to_dict()
    )
    rows: List[Dict[str, object]] = []
    for ticker, group in articles.groupby("Ticker", sort=True):
        active_days = _active_day_count(constituents, ticker)
        rows.append(
            {
                "ticker": ticker,
                "company_name": company_names.get(ticker, ""),
                "article_count": int(len(group)),
                "arabic_article_count": int((group["detected_language"] == "ar").sum()),
                "english_article_count": int((group["detected_language"] == "en").sum()),
                "first_article_date": _iso_date(group["Date"].min()),
                "last_article_date": _iso_date(group["Date"].max()),
                "active_egx30_days": int(active_days),
                "articles_per_active_trading_day": float(len(group) / active_days) if active_days else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("article_count", ascending=False).reset_index(drop=True)


def yearly_coverage(scoped_articles: pd.DataFrame, trading_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    articles = scoped_articles.copy()
    articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce").dt.normalize()
    articles["year"] = articles["Date"].dt.year
    daily_counts = articles.groupby("Date").size().rename("article_count")
    trading = pd.DataFrame({"Date": pd.to_datetime(list(trading_dates), errors="coerce")}).dropna()
    trading["Date"] = trading["Date"].dt.normalize()
    trading["year"] = trading["Date"].dt.year
    trading["article_count"] = trading["Date"].map(daily_counts).fillna(0).astype(int)

    rows: List[Dict[str, object]] = []
    for year, days in trading.groupby("year", sort=True):
        article_year = articles[articles["year"] == year]
        trading_days = int(len(days))
        covered_1 = int((days["article_count"] >= 1).sum())
        rows.append(
            {
                "year": int(year),
                "article_count": int(len(article_year)),
                "arabic_article_count": int((article_year["detected_language"] == "ar").sum()),
                "english_article_count": int((article_year["detected_language"] == "en").sum()),
                "trading_days": trading_days,
                "trading_days_with_at_least_1_article": covered_1,
                "trading_days_with_at_least_3_articles": int((days["article_count"] >= 3).sum()),
                "trading_days_with_at_least_5_articles": int((days["article_count"] >= 5).sum()),
                "covered_trading_day_pct": float(covered_1 / trading_days) if trading_days else math.nan,
            }
        )
    return pd.DataFrame(rows)


def daily_coverage(scoped_articles: pd.DataFrame, trading_dates: Iterable[pd.Timestamp], sector_map: pd.DataFrame | None = None) -> pd.DataFrame:
    articles = scoped_articles.copy()
    articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce").dt.normalize()
    articles["Ticker"] = articles["Ticker"].astype(str).str.upper()
    if sector_map is not None and not sector_map.empty:
        sector_lookup = sector_map.set_index("ticker")["sector"].to_dict()
        articles["sector"] = articles["Ticker"].map(sector_lookup).fillna("UNKNOWN")
    else:
        articles["sector"] = "UNKNOWN"
    rows: List[Dict[str, object]] = []
    for date in pd.to_datetime(list(trading_dates), errors="coerce"):
        date = pd.Timestamp(date).normalize()
        group = articles[articles["Date"] == date]
        rows.append(
            {
                "date": date.date().isoformat(),
                "article_count": int(len(group)),
                "distinct_companies_covered": int(group["Ticker"].nunique()),
                "arabic_article_count": int((group["detected_language"] == "ar").sum()),
                "english_article_count": int((group["detected_language"] == "en").sum()),
                "sector_coverage_count": int(group.loc[group["sector"] != "UNKNOWN", "sector"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def write_coverage_report(
    company: pd.DataFrame,
    yearly: pd.DataFrame,
    daily: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    company.to_csv(output_dir / "egx30_company_coverage.csv", index=False)
    yearly.to_csv(output_dir / "egx30_yearly_coverage.csv", index=False)
    daily.to_csv(output_dir / "egx30_daily_coverage.csv", index=False)
    total_articles = int(company["article_count"].sum()) if not company.empty else 0
    top5_share = float(company.head(5)["article_count"].sum() / total_articles) if total_articles else math.nan
    top10_share = float(company.head(10)["article_count"].sum() / total_articles) if total_articles else math.nan
    shares = company["article_count"] / total_articles if total_articles else pd.Series(dtype=float)
    hhi = float((shares**2).sum()) if total_articles else math.nan
    nonzero = daily[daily["article_count"] > 0]
    one_company_share = float((nonzero["distinct_companies_covered"] <= 1).mean()) if len(nonzero) else math.nan
    warnings = []
    if top5_share > 0.50:
        warnings.append("Top five companies represent more than 50% of articles.")
    if one_company_share > 0.30:
        warnings.append("More than 30% of article-covered trading days cover only one company.")
    if not warnings:
        warnings.append("No concentration warning crossed the pre-set thresholds.")

    report = f"""# EGX30 Coverage Audit

## Interpretation

This audit checks whether the point-in-time EGX30 news corpus is broadly distributed across companies and trading dates. It is an audit artifact only; EGX30 prices are used only for trading-day calendars and target labels elsewhere, not as sentiment-model features.

## Concentration Diagnostics

- Total scoped articles: `{total_articles}`
- Top 5 company article share: `{top5_share:.3f}`
- Top 10 company article share: `{top10_share:.3f}`
- Article-count HHI: `{hhi:.3f}`
- Covered trading days with only one company represented: `{one_company_share:.3f}`

## Coverage Warnings

{chr(10).join(f"- {warning}" for warning in warnings)}

## Top Companies By Article Count

{markdown_table(company, ["ticker", "company_name", "article_count", "arabic_article_count", "english_article_count", "articles_per_active_trading_day"], 15)}

## Yearly Coverage

{markdown_table(yearly, yearly.columns, 10)}

## Limitations

Coverage remains uneven when a small group of firms or sectors receives disproportionate news attention. Trading-day coverage does not establish that articles were equally informative across firms, and low-coverage days should be interpreted cautiously in forecasting results.
"""
    (output_dir / "egx30_coverage_audit.md").write_text(report, encoding="utf-8")
    return {
        "total_articles": total_articles,
        "top5_share": top5_share,
        "top10_share": top10_share,
        "hhi": hhi,
        "one_company_covered_day_share": one_company_share,
        "warnings": warnings,
    }


def baseline_rows_for_panel(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    horizon: int,
    sentiment_source: str,
    view: str,
    selected_c: float,
    random_seed: int,
    refit_every_rows: int,
) -> pd.DataFrame:
    split = infer_split(panel, predictions, horizon)
    rows: List[Dict[str, object]] = []
    for split_label in ("validation", "test"):
        eval_indices = split.validation_indices if split_label == "validation" else split.test_indices
        actual = _actual_predictions(panel, eval_indices, split_label)
        baselines = {
            "always_up": actual.assign(predicted_direction=1, predicted_probability=1.0),
            "always_down": actual.assign(predicted_direction=0, predicted_probability=0.0),
            "training_majority": _training_majority_predictions(panel, eval_indices, split.train_end, split_label),
            "rolling_majority": _rolling_majority_predictions(panel, eval_indices, horizon, split_label),
        }
        count_columns = [
            column
            for column in panel.columns
            if _is_count_feature(column) and column not in {"target_direction", "target_return"}
        ]
        if count_columns:
            baselines["article_count_only_logit"] = _walk_forward_with_selected_c(
                panel, count_columns, split, split_label, horizon, selected_c, random_seed, refit_every_rows
            )
        sentiment_columns = [
            column
            for column in panel.columns
            if _is_sentiment_feature(column) and column not in {"target_direction", "target_return"}
        ]
        if sentiment_columns:
            shuffled = panel.copy()
            rng = np.random.default_rng(random_seed + int(horizon))
            for column in sentiment_columns:
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            feature_columns = _feature_columns(shuffled)
            baselines["sentiment_shuffled_logit"] = _walk_forward_with_selected_c(
                shuffled, feature_columns, split, split_label, horizon, selected_c, random_seed, refit_every_rows
            )

        base_metrics = compute_direction_metrics(baselines["training_majority"])
        for name, frame in baselines.items():
            metrics = compute_direction_metrics(frame)
            rows.append(
                {
                    "horizon": int(horizon),
                    "sentiment_source": sentiment_source,
                    "view": view,
                    "split": split_label,
                    "baseline_name": name,
                    **metrics,
                    "delta_balanced_accuracy_vs_training_majority": _delta(
                        metrics["balanced_accuracy"], base_metrics["balanced_accuracy"]
                    ),
                    "delta_f1_vs_training_majority": _delta(metrics["f1"], base_metrics["f1"]),
                }
            )
        model_subset = predictions[
            (predictions["split"] == split_label)
            & (predictions["horizon"] == horizon)
            & (predictions["sentiment_source"] == sentiment_source)
            & (predictions["view"] == view)
        ]
        if not model_subset.empty:
            model_metrics = compute_direction_metrics(model_subset)
            rows.append(
                {
                    "horizon": int(horizon),
                    "sentiment_source": sentiment_source,
                    "view": view,
                    "split": split_label,
                    "baseline_name": "sentiment_model",
                    **model_metrics,
                    "delta_balanced_accuracy_vs_training_majority": _delta(
                        model_metrics["balanced_accuracy"], base_metrics["balanced_accuracy"]
                    ),
                    "delta_f1_vs_training_majority": _delta(model_metrics["f1"], base_metrics["f1"]),
                }
            )
    return pd.DataFrame(rows).reindex(columns=REPORT_COLUMNS)


def write_baseline_report(frame: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "egx30_baseline_comparison.csv", index=False)
    test = frame[frame["split"] == "test"].copy()
    sentiment = test[test["baseline_name"] == "sentiment_model"].copy()
    majority_beats = int((sentiment["delta_balanced_accuracy_vs_training_majority"] > 0).sum())
    total_model_rows = int(len(sentiment))
    best = sentiment.sort_values(["balanced_accuracy", "f1"], ascending=False).head(10)
    suspicious = add_metric_warnings(sentiment)
    report = f"""# EGX30 Baseline Comparison

## Interpretation

Raw accuracy is not enough for this task because EGX30 direction labels can be imbalanced. Balanced accuracy, F1, MCC, predicted Up share, and deltas versus the training-majority baseline are reported alongside accuracy.

The sentiment model beats the training-majority baseline on balanced accuracy in `{majority_beats}` of `{total_model_rows}` test rows. Rows that do not beat this baseline should be treated as weak or non-evidential even when raw accuracy is high.

## Best Sentiment-Model Test Rows

{markdown_table(best, ["horizon", "sentiment_source", "view", "accuracy", "balanced_accuracy", "f1", "mcc", "predicted_up_share", "actual_up_share", "delta_balanced_accuracy_vs_training_majority"], 10)}

## Class-Imbalance Warnings

{markdown_table(suspicious, ["horizon", "sentiment_source", "view", "accuracy", "balanced_accuracy", "f1", "predicted_up_share", "actual_up_share", "warning"], 20)}

## Baseline Notes

- `always_up` and `always_down` expose label skew.
- `training_majority` uses only the initial training split.
- `rolling_majority` uses only prior labels available before each predicted row.
- `article_count_only_logit` tests whether lagged news volume and coverage explain performance without sentiment polarity.
- `sentiment_shuffled_logit` is deterministic and breaks sentiment timing while preserving the model path.
"""
    (output_dir / "egx30_baseline_comparison.md").write_text(report, encoding="utf-8")
    return {"sentiment_rows_beating_training_majority": majority_beats, "sentiment_test_rows": total_model_rows}


def abstention_rows_for_panel(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    horizon: int,
    sentiment_source: str,
    view: str,
) -> pd.DataFrame:
    split = infer_split(panel, predictions, horizon)
    subset = predictions[
        (predictions["horizon"] == horizon)
        & (predictions["sentiment_source"] == sentiment_source)
        & (predictions["view"] == view)
        & (predictions["split"].isin(["validation", "test"]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    signal = _signal_strength(panel)
    subset["signal_strength"] = subset["row_index"].astype(int).map(signal)
    validation = subset[subset["split"] == "validation"].dropna(subset=["signal_strength"])
    test = subset[subset["split"] == "test"].dropna(subset=["signal_strength"])
    rows: List[Dict[str, object]] = []
    for coverage_target in (0.50, 0.30, 0.20):
        if validation.empty or test.empty:
            continue
        threshold = float(validation["signal_strength"].quantile(1.0 - coverage_target))
        covered = test[test["signal_strength"] >= threshold]
        metrics = compute_direction_metrics(covered)
        rows.append(
            {
                "horizon": int(horizon),
                "sentiment_source": sentiment_source,
                "view": view,
                "signal_definition": "max(abs(lagged combined sentiment), scaled article count, coverage ratio, model confidence)",
                "target_coverage": coverage_target,
                "training_selected_threshold": threshold,
                "coverage_pct": float(len(covered) / len(test)) if len(test) else math.nan,
                "covered_test_samples": int(len(covered)),
                "abstained_test_samples": int(len(test) - len(covered)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def write_abstention_report(frame: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "egx30_abstention_experiment.csv", index=False)
    best = frame.sort_values(["balanced_accuracy", "coverage_pct"], ascending=False).head(10)
    report = f"""# EGX30 Abstention Experiment

## Interpretation

This experiment asks whether sentiment-only predictions are more useful on high-signal days. Thresholds are selected from validation rows only, then applied to the test rows. Accuracy is always reported with coverage because high accuracy on limited coverage is not sufficient evidence.

## Best Covered Test Rows

{markdown_table(best, ["horizon", "sentiment_source", "view", "target_coverage", "coverage_pct", "covered_test_samples", "accuracy", "balanced_accuracy", "f1", "mcc"], 10)}

## Limitation

Abstention changes the decision problem: it evaluates fewer market sessions. These results are promising only when balanced accuracy improves while retaining enough covered samples to be practically meaningful.
"""
    (output_dir / "egx30_abstention_experiment.md").write_text(report, encoding="utf-8")
    return {"rows": int(len(frame))}


def build_sector_daily(scoped_articles: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    articles = scoped_articles.copy()
    articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce").dt.normalize()
    articles["Ticker"] = articles["Ticker"].astype(str).str.upper()
    lookup = sector_map.set_index("ticker")["sector"].to_dict()
    articles["sector"] = articles["Ticker"].map(lookup).fillna("UNKNOWN")
    score = pd.to_numeric(articles.get("continuous_sentiment_score", articles.get("sentiment_score", 0.0)), errors="coerce").fillna(0.0)
    articles["sentiment_value"] = score
    confidence = pd.to_numeric(articles.get("confidence", 1.0), errors="coerce").fillna(1.0)
    articles["confidence_value"] = confidence
    rows: List[Dict[str, object]] = []
    for (date, sector), group in articles.groupby(["Date", "sector"], sort=True):
        values = group["sentiment_value"].astype(float)
        weights = group["confidence_value"].astype(float).clip(lower=0.0)
        weighted = float(np.average(values, weights=weights)) if float(weights.sum()) > 0 else float(values.mean())
        rows.append(
            {
                "Date": date,
                "sector": sector,
                "sector_article_count": int(len(group)),
                "sector_avg_sentiment": float(values.mean()) if len(values) else 0.0,
                "sector_confidence_weighted_sentiment": weighted,
            }
        )
    return pd.DataFrame(rows)


def write_sector_report(sector_daily: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        sector_daily.groupby("sector", as_index=False)
        .agg(
            trading_days_with_articles=("Date", "nunique"),
            article_count=("sector_article_count", "sum"),
            avg_sector_sentiment=("sector_avg_sentiment", "mean"),
        )
        .sort_values("article_count", ascending=False)
        if not sector_daily.empty
        else pd.DataFrame(columns=["sector", "trading_days_with_articles", "article_count", "avg_sector_sentiment"])
    )
    summary.to_csv(output_dir / "egx30_sector_feature_ablation.csv", index=False)
    report = f"""# EGX30 Sector-Aware Aggregation

## Result

A compact sector map was added and sector-level daily aggregates were generated. This pass reports the sector feature layer as an audit-ready ablation scaffold; the main sentiment model remains unchanged.

## Sector Coverage Summary

{markdown_table(summary, summary.columns, 20)}

## Limitation

Sector labels are manually reviewable metadata, and `UNKNOWN` is tolerated by the feature builder. No sector label is used to create target labels or price-derived predictors.
"""
    (output_dir / "egx30_sector_feature_ablation.md").write_text(report, encoding="utf-8")
    return {"sectors": int(summary["sector"].nunique()) if "sector" in summary else 0}


def write_weighting_report(scoped_articles: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = scoped_articles.copy()
    articles["Date"] = pd.to_datetime(articles["Date"], errors="coerce").dt.normalize()
    score = pd.to_numeric(articles.get("continuous_sentiment_score", articles.get("sentiment_score", 0.0)), errors="coerce").fillna(0.0)
    articles["sentiment_value"] = score
    rows = []
    for date, group in articles.groupby("Date", sort=True):
        company_means = group.groupby("Ticker")["sentiment_value"].mean()
        rows.append(
            {
                "date": date.date().isoformat(),
                "equal_weighted_sentiment": float(company_means.mean()) if len(company_means) else 0.0,
                "article_count_weighted_sentiment": float(group["sentiment_value"].mean()) if len(group) else 0.0,
                "covered_companies": int(group["Ticker"].nunique()),
                "article_count": int(len(group)),
                "index_weighted_sentiment": math.nan,
                "index_weight_status": "unavailable",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "egx30_weighting_ablation.csv", index=False)
    corr = (
        float(frame["equal_weighted_sentiment"].corr(frame["article_count_weighted_sentiment"]))
        if len(frame) > 1
        else math.nan
    )
    report = f"""# EGX30 Weighting Ablation

## Result

Equal-weighted and article-count-weighted daily sentiment series were generated from the existing corpus. Historical constituent weights were not available in the current reproducible dataset, so true index-weighted sentiment was not reported.

- Equal vs article-count weighted daily sentiment correlation: `{corr:.3f}`
- True historical index weights: unavailable

No market-cap, price, return, or OHLCV proxy was used as a substitute for missing index weights.
"""
    (output_dir / "egx30_weighting_ablation.md").write_text(report, encoding="utf-8")
    return {"equal_article_weight_corr": corr, "index_weights_available": False}


def add_metric_warnings(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        warnings = []
        if row.get("predicted_up_share", 0.5) > 0.90 or row.get("predicted_up_share", 0.5) < 0.10:
            warnings.append("extreme predicted Up share")
        if 0.48 <= row.get("balanced_accuracy", math.nan) <= 0.52 and row.get("accuracy", 0.0) >= row.get(
            "majority_baseline_accuracy", 1.0
        ):
            warnings.append("near-random balanced accuracy despite high raw accuracy")
        if row.get("rows", 0) < 50:
            warnings.append("small test sample")
        if warnings:
            row["warning"] = "; ".join(warnings)
            rows.append(row)
    return pd.DataFrame(rows)


def infer_split(panel: pd.DataFrame, predictions: pd.DataFrame, horizon: int) -> TimeSeriesSplit:
    split = None
    try:
        config = SentimentOnlyPipelineConfig(tickers=(EGX30_TICKER,), horizon_days=int(horizon))
        split = build_time_series_split(len(panel), config.split_config, int(horizon))
    except ValueError:
        split = None
    subset = predictions[predictions["horizon"] == horizon]
    if not subset.empty:
        validation = subset[subset["split"] == "validation"]["row_index"].astype(int)
        test = subset[subset["split"] == "test"]["row_index"].astype(int)
        if len(validation) and len(test):
            return TimeSeriesSplit(
                train_end=split.train_end if split is not None else max(1, int(validation.min()) - max(int(horizon), 1)),
                validation_start=int(validation.min()),
                validation_end=int(validation.max()) + 1,
                test_start=int(test.min()),
                n_rows=len(panel),
            )
    if split is None:
        raise ValueError(f"Cannot infer time-series split for n_rows={len(panel)} horizon={horizon}")
    return split


def _actual_predictions(panel: pd.DataFrame, indices: Sequence[int], split_label: str) -> pd.DataFrame:
    rows = panel.iloc[list(indices)].copy()
    return pd.DataFrame(
        {
            "Date": rows["Date"].to_numpy(),
            "Ticker": rows["Ticker"].to_numpy(),
            "split": split_label,
            "row_index": rows.index.to_numpy(),
            "actual_direction": rows["target_direction"].astype(int).to_numpy(),
            "actual_return": rows["target_return"].astype(float).to_numpy(),
        }
    )


def _training_majority_predictions(panel: pd.DataFrame, indices: Sequence[int], train_end: int, split_label: str) -> pd.DataFrame:
    base = _actual_predictions(panel, indices, split_label)
    train = panel.iloc[:train_end]["target_direction"].astype(int)
    prediction = int(float(train.mean()) >= 0.5) if len(train) else 1
    return base.assign(predicted_direction=prediction, predicted_probability=float(train.mean()) if len(train) else 0.5)


def _rolling_majority_predictions(panel: pd.DataFrame, indices: Sequence[int], horizon: int, split_label: str, window: int = 60) -> pd.DataFrame:
    base = _actual_predictions(panel, indices, split_label)
    preds = []
    probs = []
    for idx in base["row_index"].astype(int):
        history_end = max(0, idx - horizon + 1)
        history = panel.iloc[max(0, history_end - window) : history_end]["target_direction"].astype(int)
        probability = float(history.mean()) if len(history) else 0.5
        preds.append(int(probability >= 0.5))
        probs.append(probability)
    return base.assign(predicted_direction=preds, predicted_probability=probs)


def _walk_forward_with_selected_c(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    split: TimeSeriesSplit,
    split_label: str,
    horizon: int,
    selected_c: float,
    random_seed: int,
    refit_every_rows: int,
) -> pd.DataFrame:
    indices = split.validation_indices if split_label == "validation" else split.test_indices
    return walk_forward_direction_predictions(
        panel=panel,
        feature_columns=feature_columns,
        evaluation_indices=indices,
        horizon_days=horizon,
        regularization_c=selected_c,
        random_seed=random_seed,
        split_label=split_label,
        refit_every_rows=refit_every_rows,
    )


def _feature_columns(panel: pd.DataFrame) -> List[str]:
    excluded = {"Date", "Ticker", "Close", "target_return", "target_direction", "target_horizon_end_date"}
    return [column for column in panel.columns if column not in excluded]


def _is_count_feature(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in ("article_count", "coverage", "covered_constituent", "active_constituent", "days_since"))


def _is_sentiment_feature(column: str) -> bool:
    lower = column.lower()
    return "sentiment" in lower or "probability" in lower or "share_of_articles" in lower


def _signal_strength(panel: pd.DataFrame) -> Dict[int, float]:
    feature = pd.Series(0.0, index=panel.index)
    candidates = [column for column in panel.columns if "combined_sentiment_weighted_mean_lag1d" in column]
    if candidates:
        feature = feature.combine(pd.to_numeric(panel[candidates[0]], errors="coerce").abs().fillna(0.0), max)
    if "combined_article_count_lag1d" in panel:
        count = pd.to_numeric(panel["combined_article_count_lag1d"], errors="coerce").fillna(0.0)
        scaled = count / max(float(count.quantile(0.95)), 1.0)
        feature = feature.combine(scaled.clip(upper=1.0), max)
    if "constituent_coverage_ratio_lag1d" in panel:
        feature = feature.combine(pd.to_numeric(panel["constituent_coverage_ratio_lag1d"], errors="coerce").fillna(0.0), max)
    return feature.astype(float).to_dict()


def _active_day_count(constituents: pd.DataFrame, ticker: str) -> int:
    frame = constituents[constituents["Ticker"].astype(str).str.upper() == ticker].copy()
    if frame.empty:
        return 0
    frame["Date_From"] = pd.to_datetime(frame["Date_From"], errors="coerce").dt.normalize()
    frame["Date_To"] = pd.to_datetime(frame["Date_To"], errors="coerce").dt.normalize()
    days = set()
    for row in frame.itertuples(index=False):
        if pd.isna(row.Date_From) or pd.isna(row.Date_To):
            continue
        days.update(pd.date_range(row.Date_From, row.Date_To, freq="B").tolist())
    return len(days)


def _iso_date(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    return timestamp.date().isoformat() if pd.notna(timestamp) else ""


def _delta(value: object, base: object) -> float:
    try:
        if math.isnan(float(value)) or math.isnan(float(base)):
            return math.nan
        return float(value) - float(base)
    except Exception:
        return math.nan
