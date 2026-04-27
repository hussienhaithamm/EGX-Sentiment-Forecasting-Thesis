"""Event-aware sentiment feature engineering for the EGX thesis pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from egx_direction.sentiment_only_pipeline import build_sentiment_targets
from news_pipeline.tickers import normalize_text


EVENT_KEYWORDS_BY_CATEGORY: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "earnings": {
        "en": ("earnings", "results", "profit", "net profit", "revenue", "sales"),
        "ar": ("أرباح", "نتائج", "صافي الربح", "الإيرادات", "المبيعات"),
    },
    "dividends": {
        "en": ("dividend",),
        "ar": ("توزيعات",),
    },
    "ma": {
        "en": ("acquisition", "merger"),
        "ar": ("استحواذ", "اندماج"),
    },
    "contracts_projects": {
        "en": ("contract", "project"),
        "ar": ("عقد", "مشروع"),
    },
    "expansion_investment": {
        "en": ("expansion", "investment"),
        "ar": ("توسع", "استثمار"),
    },
    "financing": {
        "en": ("financing", "loan"),
        "ar": ("تمويل", "قرض"),
    },
    "governance_disclosure": {
        "en": ("board", "disclosure", "financial statements"),
        "ar": ("مجلس الإدارة", "إفصاح", "القوائم المالية"),
    },
}

EVENT_CATEGORY_ORDER = tuple(EVENT_KEYWORDS_BY_CATEGORY.keys())


def _article_event_text(row: pd.Series) -> str:
    parts: List[str] = []
    for column in (
        "clean_article_text",
        "cleaned_title",
        "cleaned_body_text",
        "article_title",
        "title",
        "article_summary",
        "article_body",
        "body_text",
    ):
        value = row.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in parts:
            parts.append(text)
    return normalize_text(" ".join(parts))


def _keyword_is_ticker_name_collision(ticker: str, keyword: str) -> bool:
    normalized_ticker = str(ticker or "").upper()
    normalized_keyword = normalize_text(keyword)
    return normalized_ticker == "OIH" and normalized_keyword in {"investment", normalize_text("استثمار")}


def _find_event_matches(normalized_text: str, ticker: str = "") -> Tuple[List[str], List[str]]:
    categories: List[str] = []
    matches: List[str] = []
    if not normalized_text:
        return categories, matches

    for category, language_keywords in EVENT_KEYWORDS_BY_CATEGORY.items():
        category_matches: List[str] = []
        for language in ("en", "ar"):
            for keyword in language_keywords[language]:
                if _keyword_is_ticker_name_collision(ticker, keyword):
                    continue
                normalized_keyword = normalize_text(keyword)
                if normalized_keyword and normalized_keyword in normalized_text:
                    category_matches.append(keyword)
        if category_matches:
            categories.append(category)
            matches.extend(f"{category}:{keyword}" for keyword in sorted(set(category_matches)))
    return categories, matches


def label_event_articles(scored_articles: pd.DataFrame) -> pd.DataFrame:
    """Add event flags, categories, keyword matches, and event weights to articles."""

    frame = scored_articles.copy()
    if frame.empty:
        for column in (
            "event_flag",
            "event_category",
            "event_categories",
            "event_keyword_matches",
            "event_keyword_match_count",
            "event_weight",
        ):
            frame[column] = []
        return frame

    event_categories: List[str] = []
    event_category_lists: List[str] = []
    event_matches: List[str] = []
    event_counts: List[int] = []
    event_weights: List[float] = []

    for _, row in frame.iterrows():
        categories, matches = _find_event_matches(_article_event_text(row), str(row.get("Ticker", "")))
        category_counts = Counter(categories)
        if categories:
            primary_category = sorted(
                category_counts,
                key=lambda item: (-category_counts[item], EVENT_CATEGORY_ORDER.index(item)),
            )[0]
        else:
            primary_category = "none"

        match_count = len(matches)
        category_count = len(set(categories))
        event_weight = 1.0
        if match_count:
            event_weight += min(1.5, 0.25 * match_count) + min(0.5, 0.15 * max(category_count - 1, 0))

        event_categories.append(primary_category)
        event_category_lists.append("|".join(categories))
        event_matches.append("|".join(matches))
        event_counts.append(match_count)
        event_weights.append(float(min(event_weight, 3.0)))

    frame["event_flag"] = np.array(event_counts, dtype=int) > 0
    frame["event_category"] = event_categories
    frame["event_categories"] = event_category_lists
    frame["event_keyword_matches"] = event_matches
    frame["event_keyword_match_count"] = event_counts
    frame["event_weight"] = event_weights
    return frame


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_values = values.fillna(0.0).astype(float).to_numpy()
    clean_weights = weights.fillna(0.0).astype(float).to_numpy()
    if len(clean_values) == 0:
        return 0.0
    if clean_weights.sum() <= 0:
        return float(clean_values.mean())
    return float(np.average(clean_values, weights=clean_weights))


def _empty_event_aggregate(prefix: str) -> Dict[str, float]:
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
        f"{prefix}_sentiment_volume": 0.0,
        f"{prefix}_event_weight_mean": 0.0,
        f"{prefix}_event_weight_total": 0.0,
        f"{prefix}_keyword_match_count": 0.0,
    }


def _event_weights(group: pd.DataFrame) -> pd.Series:
    base = group.get("sentiment_article_weight", 1.0)
    base_weights = pd.to_numeric(base, errors="coerce").fillna(0.0).astype(float)
    boost = pd.to_numeric(group.get("event_weight", 1.0), errors="coerce").fillna(1.0).astype(float)
    return base_weights * boost


def _aggregate_subset(group: pd.DataFrame, prefix: str) -> Dict[str, float]:
    row = _empty_event_aggregate(prefix)
    if group.empty:
        return row

    scored = group[group["is_scored"] == 1].copy()
    row[f"{prefix}_article_count"] = float(len(group))
    row[f"{prefix}_scored_count"] = float(len(scored))
    row[f"{prefix}_event_weight_mean"] = float(pd.to_numeric(group["event_weight"], errors="coerce").fillna(1.0).mean())
    row[f"{prefix}_event_weight_total"] = float(pd.to_numeric(group["event_weight"], errors="coerce").fillna(1.0).sum())
    row[f"{prefix}_keyword_match_count"] = float(
        pd.to_numeric(group["event_keyword_match_count"], errors="coerce").fillna(0.0).sum()
    )
    if scored.empty:
        return row

    scores = scored["sentiment_score_standardized"].fillna(0.0).astype(float)
    weights = _event_weights(scored)
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
            f"{prefix}_sentiment_volume": float((scores * weights).sum()),
        }
    )
    return row


def _language_view_mask(frame: pd.DataFrame, view: str) -> pd.Series:
    if view == "english_only":
        return frame["detected_language"].eq("en")
    if view == "arabic_only":
        return frame["detected_language"].eq("ar")
    if view == "bilingual":
        return frame["detected_language"].isin({"en", "ar"})
    raise ValueError(f"Unsupported view: {view}")


def aggregate_daily_event_features(scored_articles: pd.DataFrame, view: str, strategy: str) -> pd.DataFrame:
    """Aggregate all-news, event-only, or event-weighted sentiment features."""

    if strategy not in {"event_only", "event_weighted"}:
        raise ValueError("strategy must be 'event_only' or 'event_weighted'.")
    if scored_articles.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])

    subset = scored_articles[_language_view_mask(scored_articles, view)].copy()
    if strategy == "event_only":
        subset = subset[subset["event_flag"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])

    rows: List[Dict[str, float]] = []
    for (ticker, date), group in subset.groupby(["Ticker", "Date"], sort=True):
        event_group = group[group["event_flag"]].copy()
        row: Dict[str, float] = {"Ticker": ticker, "Date": date}
        row.update(_aggregate_subset(group, "combined"))
        row.update(_aggregate_subset(group[group["detected_language"] == "en"], "en"))
        row.update(_aggregate_subset(group[group["detected_language"] == "ar"], "ar"))
        row.update(_aggregate_subset(event_group, "event"))
        for category in EVENT_CATEGORY_ORDER:
            row[f"event_{category}_count"] = float((event_group["event_category"] == category).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _is_sum_like(column: str) -> bool:
    return any(
        token in column
        for token in (
            "_count",
            "_total",
            "_volume",
            "_sum",
            "_flag",
            "_matches",
            "_match",
        )
    )


def _days_since_event(series: pd.Series) -> pd.Series:
    values: List[float] = []
    seen = None
    default = max(len(series), 1)
    for flag in series.fillna(0).astype(float).gt(0).tolist():
        if flag:
            seen = 0
        elif seen is None:
            seen = default
        else:
            seen += 1
        values.append(float(seen))
    return pd.Series(values, index=series.index, dtype=float).clip(upper=365.0)


def align_event_features_to_prices(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    rolling_windows: Sequence[int],
    lag_days: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Align event features to trading dates using only prior calendar information."""

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if prices.empty:
        return pd.DataFrame(columns=["Date", "Ticker"])

    ticker = str(prices["Ticker"].iloc[0])
    price_dates = pd.Index(prices["Date"])
    if daily_features.empty:
        base_columns = ["event_article_count", "event_sentiment_weighted_mean"]
        base_daily = pd.DataFrame(index=pd.date_range(price_dates.min(), price_dates.max(), freq="D"))
        for column in base_columns:
            base_daily[column] = 0.0
    else:
        daily = daily_features.copy()
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
        base_columns = [column for column in daily.columns if column not in {"Ticker", "Date"}]
        start_date = min(price_dates.min(), daily["Date"].min())
        calendar_index = pd.date_range(start_date, price_dates.max(), freq="D")
        base_daily = daily.set_index("Date")[base_columns].reindex(calendar_index).fillna(0.0)

    feature_data: Dict[str, pd.Series] = {}
    for lag in sorted(set(int(value) for value in lag_days)):
        shifted = base_daily.shift(lag).fillna(0.0)
        for column in base_columns:
            feature_data[f"{column}_lag{lag}d"] = shifted[column]
            if lag == 1:
                feature_data[column] = shifted[column]

    shifted_one = base_daily.shift(1).fillna(0.0)
    for column in base_columns:
        series = shifted_one[column]
        for window in sorted(set(int(value) for value in rolling_windows)):
            if _is_sum_like(column):
                feature_data[f"{column}_roll_sum_{window}d"] = series.rolling(window, min_periods=1).sum().fillna(0.0)
            else:
                feature_data[f"{column}_roll_mean_{window}d"] = series.rolling(window, min_periods=1).mean().fillna(0.0)
                feature_data[f"{column}_roll_std_{window}d"] = series.rolling(window, min_periods=2).std(ddof=0).fillna(0.0)

    if "event_article_count" in shifted_one.columns:
        feature_data["days_since_event_article"] = _days_since_event(shifted_one["event_article_count"])
    if "combined_article_count" in shifted_one.columns:
        feature_data["days_since_combined_article"] = _days_since_event(shifted_one["combined_article_count"])

    aligned = pd.DataFrame(feature_data, index=base_daily.index)
    aligned = aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=-1000.0, upper=1000.0)
    aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": "Date"})
    aligned["Ticker"] = ticker
    return aligned[["Date", "Ticker"] + [column for column in aligned.columns if column not in {"Date", "Ticker"}]]


def build_event_feature_panel(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    horizon_days: int,
    rolling_windows: Sequence[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge event-aware sentiment features with future return/direction targets."""

    aligned_features = align_event_features_to_prices(
        price_df=price_df,
        daily_features=daily_features,
        rolling_windows=rolling_windows,
    )
    targets = build_sentiment_targets(price_df, horizon_days=horizon_days)
    panel = targets.merge(aligned_features, how="left", on=["Date", "Ticker"])
    excluded = {"Date", "Ticker", "Close", "target_return", "target_direction", "target_horizon_end_date"}
    feature_columns = [column for column in panel.columns if column not in excluded]
    panel[feature_columns] = panel[feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(float)
    panel[feature_columns] = panel[feature_columns].clip(lower=-1000.0, upper=1000.0)
    panel["target_direction"] = panel["target_direction"].astype(int)
    return panel.sort_values("Date").reset_index(drop=True), feature_columns
