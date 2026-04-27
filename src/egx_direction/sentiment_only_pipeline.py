"""Generalized sentiment-only direction and return pipeline for EGX tickers."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import math
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import StockDataLoader
from news_pipeline import (
    AlMobasherScraper,
    AlMobasherScraperConfig,
    FinBERTConfig,
    FinBERTSentimentNewsFeatureBuilder,
    NewsFeatureConfig,
    resolve_ticker,
)
from news_pipeline.tickers import TARGET_COMPANIES, normalize_text


LANGUAGE_VIEWS = ("english_only", "arabic_only", "bilingual")

_CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

_MATCH_CONFIDENCE_WEIGHT = {
    "low": 0.60,
    "medium": 0.80,
    "high": 1.00,
}

_RELEVANCE_LABEL_WEIGHT = {
    "direct": 1.00,
    "contextual": 0.85,
    "indirect": 0.65,
    "stock_page_only": 0.45,
}

_ARTICLE_PREPARED_COLUMNS = [
    "ticker",
    "canonical_ticker",
    "Ticker",
    "Date",
    "company_name",
    "company_name_en",
    "company_name_ar",
    "source",
    "language",
    "published_at",
    "published_timestamp",
    "scraped_timestamp",
    "article_id",
    "article_url",
    "url",
    "article_language",
    "match_confidence",
    "match_methods",
    "matched_primary_aliases",
    "matched_secondary_aliases",
    "matched_aliases",
    "matched_keywords",
    "entity_relevance_label",
    "entity_relevance_weight",
    "relevance_flag",
    "relevance_score",
    "article_title",
    "title",
    "article_summary",
    "article_body",
    "body_text",
    "cleaned_title",
    "cleaned_summary",
    "cleaned_body_text",
    "clean_article_text",
    "duplicate_flag",
    "deduplication_id",
    "near_deduplication_id",
    "is_paywalled",
]

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }
)


@dataclass(frozen=True)
class SplitConfig:
    """Time-series split config for validation and final test evaluation."""

    validation_ratio: float = 0.20
    test_ratio: float = 0.20
    min_train_rows: int = 250
    embargo_rows: int = 5

    def __post_init__(self) -> None:
        if self.validation_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("validation_ratio and test_ratio must both be positive.")
        if self.validation_ratio + self.test_ratio >= 1.0:
            raise ValueError("validation_ratio + test_ratio must be < 1.0.")
        if self.min_train_rows < 50:
            raise ValueError("min_train_rows must be at least 50.")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows must be >= 0.")


@dataclass(frozen=True)
class TimeSeriesSplit:
    """Concrete row boundaries for one ticker's panel."""

    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    n_rows: int

    @property
    def validation_indices(self) -> np.ndarray:
        return np.arange(self.validation_start, self.validation_end)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_start, self.n_rows)


@dataclass(frozen=True)
class SentimentOnlyPipelineConfig:
    """Runtime config for the end-to-end sentiment-only benchmark."""

    tickers: Tuple[str, ...] = ("CIB", "TMG", "OIH")
    news_data_dir: Path = Path("data/raw/news/mubasher_sentiment_only")
    processed_news_dir: Path = Path("data/processed/news/sentiment_ready")
    output_dir: Path = Path("outputs/sentiment_only_pipeline")
    scrape_languages: Tuple[str, ...] = ("en", "ar")
    scrape_max_pages: int = 100
    force_scrape: bool = False
    price_period: str = "10y"
    timezone: str = "Africa/Cairo"
    lag_strategy: str = "previous_day"
    min_match_confidence: str = "medium"
    horizon_days: int = 5
    rolling_windows: Tuple[int, ...] = (3, 7, 14)
    random_seed: int = 42
    refit_every_rows: int = 5
    split_config: SplitConfig = field(default_factory=SplitConfig)
    classification_c_grid: Tuple[float, ...] = (0.1, 1.0)
    regression_alpha_grid: Tuple[float, ...] = (1.0, 10.0)
    finbert_config: FinBERTConfig = field(default_factory=FinBERTConfig)

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1.")
        if not self.rolling_windows:
            raise ValueError("rolling_windows must contain at least one window.")
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling_windows must contain only positive integers.")
        if self.min_match_confidence not in _CONFIDENCE_RANK:
            raise ValueError("min_match_confidence must be one of: low, medium, high.")
        if self.lag_strategy != "previous_day":
            raise ValueError("This benchmark only supports lag_strategy='previous_day'.")
        if self.refit_every_rows < 1:
            raise ValueError("refit_every_rows must be >= 1.")

    @property
    def news_feature_config(self) -> NewsFeatureConfig:
        return NewsFeatureConfig(
            news_data_dir=self.news_data_dir,
            timezone=self.timezone,
            lag_strategy=self.lag_strategy,
            rolling_windows=self.rolling_windows,
            min_match_confidence=self.min_match_confidence,
        )


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(_ARABIC_NORMALIZATION_MAP)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _compose_article_text(row: pd.Series) -> str:
    parts: List[str] = []
    for column in ("cleaned_title", "cleaned_summary", "cleaned_body_text", "article_title", "article_summary", "article_body"):
        text = _clean_text(row.get(column, ""))
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts)


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _dedup_signature(row: pd.Series) -> str:
    text = normalize_text(
        " ".join(
            [
                str(row.get("cleaned_title", "") or ""),
                str(row.get("cleaned_summary", "") or ""),
                str(row.get("cleaned_body_text", "") or ""),
            ]
        )
    )
    if not text:
        text = normalize_text(str(row.get("article_url", "") or row.get("url", "") or row.get("article_id", "")))
    return _stable_hash(text)


def _near_dedup_signature(row: pd.Series) -> str:
    title = normalize_text(str(row.get("cleaned_title", "") or ""))
    if len(title) >= 24:
        return _stable_hash(title)
    text = normalize_text(str(row.get("clean_article_text", "") or ""))
    if not text:
        return _dedup_signature(row)
    tokens = text.split()
    return _stable_hash(" ".join(tokens[:160]))


def _company_metadata(ticker: str, language: str) -> Tuple[str, str, str]:
    company = TARGET_COMPANIES.get(ticker)
    if company is None:
        return "", "", ""
    company_name = company.company_name_ar if language == "ar" else company.company_name_en
    return company_name, company.company_name_en, company.company_name_ar


def _combine_aliases(row: pd.Series) -> List[str]:
    return sorted(set(_coerce_list(row.get("matched_primary_aliases")) + _coerce_list(row.get("matched_secondary_aliases"))))


def _model_route_for_language(language: object) -> str:
    value = str(language or "").lower()
    if value == "en":
        return "FinBERT"
    if value == "ar":
        return "AraBERT"
    return "unscored"


def _coerce_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith(("[", "(")) and text.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def _resolve_relevance(row: pd.Series) -> Tuple[str, float]:
    methods = set(_coerce_list(row.get("match_methods")))
    primary_hits = _coerce_list(row.get("matched_primary_aliases"))
    secondary_hits = _coerce_list(row.get("matched_secondary_aliases"))
    match_confidence = str(row.get("match_confidence", "low") or "low").lower()

    if "title_alias" in methods and primary_hits:
        label = "direct"
    elif "body_alias" in methods and primary_hits:
        label = "contextual"
    elif secondary_hits or match_confidence == "medium":
        label = "indirect"
    else:
        label = "stock_page_only"

    weight = _RELEVANCE_LABEL_WEIGHT[label] * _MATCH_CONFIDENCE_WEIGHT.get(match_confidence, 0.60)
    return label, float(min(max(weight, 0.0), 1.0))


def prepare_sentiment_articles(
    raw_news: pd.DataFrame,
    timezone: str,
    min_match_confidence: str,
) -> pd.DataFrame:
    """Normalize, filter, and clean raw news records for sentiment scoring."""

    if raw_news.empty:
        return pd.DataFrame(columns=_ARTICLE_PREPARED_COLUMNS)

    frame = raw_news.copy()
    raw_ticker = frame.get("ticker", frame.get("Ticker"))
    frame["Ticker"] = raw_ticker.map(resolve_ticker)
    frame["canonical_ticker"] = frame["Ticker"]
    frame["ticker"] = frame["Ticker"]
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["article_language"] = frame["article_language"].fillna("").astype(str).str.strip().str.lower()
    frame["language"] = frame["article_language"]
    frame["match_confidence"] = frame.get("match_confidence", "low").fillna("low").astype(str).str.lower()
    minimum_rank = _CONFIDENCE_RANK[min_match_confidence]
    frame = frame[
        frame["match_confidence"].map(lambda value: _CONFIDENCE_RANK.get(str(value).lower(), -1)) >= minimum_rank
    ].copy()

    frame = frame.dropna(subset=["Ticker", "published_at"])
    frame = frame.sort_values(["Ticker", "published_at", "article_id"], ascending=[True, True, True]).reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["Ticker", "article_language", "article_id"], keep="last").copy()

    frame["article_title"] = frame["article_title"].map(_clean_text)
    frame["article_summary"] = frame["article_summary"].map(_clean_text)
    frame["article_body"] = frame["article_body"].map(_clean_text)
    frame["cleaned_title"] = frame["article_title"]
    frame["cleaned_summary"] = frame["article_summary"]
    frame["cleaned_body_text"] = frame["article_body"]
    frame["clean_article_text"] = frame.apply(_compose_article_text, axis=1)
    frame = frame[frame["clean_article_text"].str.len() >= 20].copy()

    frame["deduplication_id"] = frame.apply(_dedup_signature, axis=1)
    frame["near_deduplication_id"] = frame.apply(_near_dedup_signature, axis=1)
    exact_duplicate_mask = frame.duplicated(
        subset=["Ticker", "article_language", "deduplication_id"],
        keep="first",
    )
    frame = frame[~exact_duplicate_mask].copy()
    near_duplicate_mask = frame.duplicated(
        subset=["Ticker", "article_language", "near_deduplication_id"],
        keep="first",
    )
    frame = frame[~near_duplicate_mask].copy()
    frame["duplicate_flag"] = False

    local_dates = frame["published_at"].dt.tz_convert(timezone).dt.tz_localize(None)
    frame["Date"] = local_dates.dt.normalize()

    relevance = frame.apply(_resolve_relevance, axis=1)
    frame["entity_relevance_label"] = relevance.map(lambda item: item[0])
    frame["entity_relevance_weight"] = relevance.map(lambda item: item[1])
    frame["relevance_flag"] = frame["entity_relevance_weight"] >= _RELEVANCE_LABEL_WEIGHT["indirect"]
    frame["relevance_score"] = frame["entity_relevance_weight"]
    frame["is_paywalled"] = frame.get("is_paywalled", False).fillna(False).astype(bool)

    frame["company_name"] = frame.apply(
        lambda row: _company_metadata(str(row["Ticker"]), str(row.get("article_language", "")))[0]
        or _clean_text(row.get("company_name", "")),
        axis=1,
    )
    frame["company_name_en"] = frame.apply(
        lambda row: _company_metadata(str(row["Ticker"]), str(row.get("article_language", "")))[1]
        or _clean_text(row.get("company_name_en", "")),
        axis=1,
    )
    frame["company_name_ar"] = frame.apply(
        lambda row: _company_metadata(str(row["Ticker"]), str(row.get("article_language", "")))[2]
        or _clean_text(row.get("company_name_ar", "")),
        axis=1,
    )
    if "source" in frame.columns:
        frame["source"] = frame["source"].fillna("Al Mobasher").astype(str)
    else:
        frame["source"] = "Al Mobasher"
    frame["published_timestamp"] = frame["published_at"]
    frame["scraped_timestamp"] = pd.to_datetime(frame.get("scraped_at", pd.NaT), utc=True, errors="coerce")
    frame["url"] = frame["article_url"]
    frame["title"] = frame["article_title"]
    frame["body_text"] = frame["article_body"]
    for column in ("match_methods", "matched_primary_aliases", "matched_secondary_aliases"):
        if column not in frame.columns:
            frame[column] = [[] for _ in range(len(frame))]
    frame["matched_aliases"] = frame.apply(_combine_aliases, axis=1)
    frame["matched_keywords"] = frame["matched_aliases"]

    ordered = frame[_ARTICLE_PREPARED_COLUMNS].sort_values(["Ticker", "Date", "published_at", "article_id"]).reset_index(drop=True)
    return ordered


def score_prepared_articles(
    prepared_articles: pd.DataFrame,
    config: SentimentOnlyPipelineConfig,
) -> pd.DataFrame:
    """Route Arabic/English articles to the correct model and standardize scores."""

    if prepared_articles.empty:
        return prepared_articles.copy()

    builder = FinBERTSentimentNewsFeatureBuilder(
        config=config.news_feature_config,
        finbert_config=FinBERTConfig(
            english_model_name=config.finbert_config.english_model_name,
            arabic_model_name=config.finbert_config.arabic_model_name,
            batch_size=config.finbert_config.batch_size,
            max_length=config.finbert_config.max_length,
            article_output_path=None,
        ),
    )
    scored = builder.score_articles(prepared_articles.copy())
    scored["sentiment_article_weight"] = (
        scored["entity_relevance_weight"].fillna(0.0).astype(float)
        * scored["confidence"].fillna(0.0).astype(float)
        * scored["is_scored"].fillna(0).astype(int)
    )
    scored["sentiment_score_standardized"] = scored["sentiment_score"].fillna(0.0).astype(float)
    scored["model_route"] = scored["detected_language"].map(_model_route_for_language)
    scored["positive_probability"] = scored["positive_prob"]
    scored["negative_probability"] = scored["negative_prob"]
    scored["neutral_probability"] = scored["neutral_prob"]
    scored["continuous_sentiment_score"] = scored["sentiment_score_standardized"]
    return scored.sort_values(["Ticker", "Date", "published_at", "article_id"]).reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return 0.0
    clean_weights = weights.fillna(0.0).astype(float).to_numpy()
    clean_values = values.fillna(0.0).astype(float).to_numpy()
    if clean_weights.sum() <= 0:
        return float(clean_values.mean())
    return float(np.average(clean_values, weights=clean_weights))


def _language_view_mask(frame: pd.DataFrame, view: str) -> pd.Series:
    if view == "english_only":
        return frame["detected_language"].eq("en")
    if view == "arabic_only":
        return frame["detected_language"].eq("ar")
    if view == "bilingual":
        return frame["detected_language"].isin({"en", "ar"})
    raise ValueError(f"Unsupported view: {view}")


def aggregate_daily_sentiment_features(
    scored_articles: pd.DataFrame,
    view: str,
) -> pd.DataFrame:
    """Aggregate article-level sentiment into one row per ticker and local date."""

    if scored_articles.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])

    subset = scored_articles[_language_view_mask(scored_articles, view)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Ticker", "Date"])

    rows: List[Dict[str, float]] = []
    for (ticker, date), group in subset.groupby(["Ticker", "Date"], sort=True):
        scored = group[group["is_scored"] == 1].copy()
        weights = scored["sentiment_article_weight"].fillna(0.0).astype(float)

        row: Dict[str, float] = {
            "Ticker": ticker,
            "Date": date,
            "article_count_total": float(len(group)),
            "article_count_scored": float(len(scored)),
            "article_count_positive": float((scored["sentiment_label"] == "positive").sum()),
            "article_count_negative": float((scored["sentiment_label"] == "negative").sum()),
            "article_count_neutral": float((scored["sentiment_label"] == "neutral").sum()),
            "article_count_direct": float((group["entity_relevance_label"] == "direct").sum()),
            "article_count_contextual": float((group["entity_relevance_label"] == "contextual").sum()),
            "article_count_indirect": float((group["entity_relevance_label"] == "indirect").sum()),
            "article_has_news": 1.0,
            "article_weight_total": float(group["entity_relevance_weight"].fillna(0.0).sum()),
            "relevance_weight_mean": float(group["entity_relevance_weight"].fillna(0.0).mean()),
        }

        if scored.empty:
            row.update(
                {
                    "model_confidence_mean": 0.0,
                    "sentiment_score_mean": 0.0,
                    "sentiment_score_weighted_mean": 0.0,
                    "sentiment_score_sum": 0.0,
                    "sentiment_score_abs_mean": 0.0,
                    "sentiment_score_std": 0.0,
                    "sentiment_score_min": 0.0,
                    "sentiment_score_max": 0.0,
                    "sentiment_positive_prob_mean": 0.0,
                    "sentiment_negative_prob_mean": 0.0,
                    "sentiment_neutral_prob_mean": 0.0,
                    "sentiment_positive_prob_weighted_mean": 0.0,
                    "sentiment_negative_prob_weighted_mean": 0.0,
                    "sentiment_neutral_prob_weighted_mean": 0.0,
                    "sentiment_positive_share": 0.0,
                    "sentiment_negative_share": 0.0,
                    "sentiment_neutral_share": 0.0,
                }
            )
        else:
            scored_en = scored[scored["detected_language"] == "en"]
            scored_ar = scored[scored["detected_language"] == "ar"]
            row.update(
                {
                    "model_confidence_mean": float(scored["confidence"].fillna(0.0).mean()),
                    "sentiment_score_mean": float(scored["sentiment_score_standardized"].mean()),
                    "sentiment_score_weighted_mean": _weighted_mean(
                        scored["sentiment_score_standardized"],
                        weights,
                    ),
                    "sentiment_score_sum": float(scored["sentiment_score_standardized"].sum()),
                    "sentiment_score_abs_mean": float(np.abs(scored["sentiment_score_standardized"]).mean()),
                    "sentiment_score_std": float(scored["sentiment_score_standardized"].std(ddof=0)),
                    "sentiment_score_min": float(scored["sentiment_score_standardized"].min()),
                    "sentiment_score_max": float(scored["sentiment_score_standardized"].max()),
                    "sentiment_positive_prob_mean": float(scored["positive_prob"].fillna(0.0).mean()),
                    "sentiment_negative_prob_mean": float(scored["negative_prob"].fillna(0.0).mean()),
                    "sentiment_neutral_prob_mean": float(scored["neutral_prob"].fillna(0.0).mean()),
                    "sentiment_positive_prob_weighted_mean": _weighted_mean(scored["positive_prob"], weights),
                    "sentiment_negative_prob_weighted_mean": _weighted_mean(scored["negative_prob"], weights),
                    "sentiment_neutral_prob_weighted_mean": _weighted_mean(scored["neutral_prob"], weights),
                    "sentiment_positive_share": float((scored["sentiment_label"] == "positive").mean()),
                    "sentiment_negative_share": float((scored["sentiment_label"] == "negative").mean()),
                    "sentiment_neutral_share": float((scored["sentiment_label"] == "neutral").mean()),
                }
            )
            if view == "bilingual":
                row["article_count_en"] = float((group["detected_language"] == "en").sum())
                row["article_count_ar"] = float((group["detected_language"] == "ar").sum())
                row["sentiment_score_mean_en"] = (
                    float(scored_en["sentiment_score_standardized"].mean()) if not scored_en.empty else 0.0
                )
                row["sentiment_score_mean_ar"] = (
                    float(scored_ar["sentiment_score_standardized"].mean()) if not scored_ar.empty else 0.0
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _sum_feature_columns(feature_columns: Iterable[str]) -> List[str]:
    columns: List[str] = []
    for column in feature_columns:
        if column.startswith("article_count_") or column.endswith("_sum") or column.endswith("_total"):
            columns.append(column)
        elif column == "article_has_news":
            columns.append(column)
    return columns


def _days_since_last_event(series: pd.Series) -> pd.Series:
    values: List[float] = []
    seen: Optional[int] = None
    max_default = max(len(series), 1)
    for flag in series.fillna(0).astype(bool).tolist():
        if flag:
            seen = 0
        elif seen is None:
            seen = max_default
        else:
            seen += 1
        values.append(float(seen))
    return pd.Series(values, index=series.index, dtype=float)


def align_sentiment_features_to_prices(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    rolling_windows: Sequence[int],
    lag_days: int = 1,
) -> pd.DataFrame:
    """Align daily calendar sentiment to the observed trading dates for one ticker."""

    if price_df.empty:
        return pd.DataFrame(columns=["Date", "Ticker"])

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    ticker = str(prices["Ticker"].iloc[0])
    price_dates = pd.Index(prices["Date"])

    if daily_features.empty:
        base_feature_columns: List[str] = ["article_count_total"]
        base_daily = pd.DataFrame(index=pd.date_range(price_dates.min(), price_dates.max(), freq="D"))
        base_daily["article_count_total"] = 0.0
    else:
        base_feature_columns = [column for column in daily_features.columns if column not in {"Ticker", "Date"}]
        start_date = min(price_dates.min(), pd.to_datetime(daily_features["Date"]).min())
        calendar_index = pd.date_range(start_date, price_dates.max(), freq="D")
        base_daily = (
            daily_features.assign(Date=pd.to_datetime(daily_features["Date"]))
            .set_index("Date")[base_feature_columns]
            .reindex(calendar_index)
            .fillna(0.0)
        )

    sum_columns = _sum_feature_columns(base_feature_columns)
    mean_columns = [column for column in base_feature_columns if column not in sum_columns]

    shifted = base_daily[base_feature_columns].shift(lag_days).fillna(0.0)
    aligned = shifted.copy()
    for window in rolling_windows:
        for column in sum_columns:
            aligned[f"{column}_roll{window}d"] = shifted[column].rolling(window=window, min_periods=1).sum().fillna(0.0)
        for column in mean_columns:
            aligned[f"{column}_roll{window}d"] = shifted[column].rolling(window=window, min_periods=1).mean().fillna(0.0)

    if "article_count_total" in shifted.columns:
        aligned["days_since_article"] = _days_since_last_event(shifted["article_count_total"] > 0).clip(upper=365.0)
    else:
        aligned["days_since_article"] = float(len(aligned))

    if "article_count_scored" in shifted.columns:
        aligned["days_since_scored_article"] = _days_since_last_event(shifted["article_count_scored"] > 0).clip(
            upper=365.0
        )
    else:
        aligned["days_since_scored_article"] = float(len(aligned))

    short_window = min(rolling_windows)
    long_window = max(rolling_windows)
    score_short = f"sentiment_score_weighted_mean_roll{short_window}d"
    score_long = f"sentiment_score_weighted_mean_roll{long_window}d"
    volume_short = f"article_count_total_roll{short_window}d"
    volume_long = f"article_count_total_roll{long_window}d"
    if score_short in aligned.columns and score_long in aligned.columns:
        aligned["sentiment_score_momentum"] = aligned[score_short] - aligned[score_long]
    if volume_short in aligned.columns and volume_long in aligned.columns:
        aligned["sentiment_volume_momentum"] = aligned[volume_short] - aligned[volume_long]

    aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": "Date"})
    aligned["Ticker"] = ticker
    ordered_columns = ["Date", "Ticker"] + [column for column in aligned.columns if column not in {"Date", "Ticker"}]
    return aligned[ordered_columns]


def build_sentiment_targets(price_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Create future direction and future return targets from trading closes."""

    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    future_close = prices["Close"].shift(-horizon_days)
    targets = prices[["Date", "Ticker", "Close"]].copy()
    targets["target_return"] = (future_close / prices["Close"]) - 1.0
    targets["target_direction"] = (targets["target_return"] > 0).astype(float)
    targets["target_horizon_end_date"] = prices["Date"].shift(-horizon_days)
    return targets.dropna(subset=["target_return"]).reset_index(drop=True)


def build_sentiment_panel(
    price_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    config: SentimentOnlyPipelineConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge aligned sentiment features with future targets for one ticker."""

    aligned_features = align_sentiment_features_to_prices(
        price_df=price_df,
        daily_features=daily_features,
        rolling_windows=config.rolling_windows,
        lag_days=1,
    )
    targets = build_sentiment_targets(price_df, horizon_days=config.horizon_days)
    panel = targets.merge(aligned_features, how="left", on=["Date", "Ticker"])
    feature_columns = [
        column
        for column in panel.columns
        if column
        not in {
            "Date",
            "Ticker",
            "Close",
            "target_return",
            "target_direction",
            "target_horizon_end_date",
        }
    ]
    panel[feature_columns] = panel[feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(float)
    panel[feature_columns] = panel[feature_columns].clip(lower=-1000.0, upper=1000.0)
    panel["target_direction"] = panel["target_direction"].astype(int)
    panel = panel.sort_values("Date").reset_index(drop=True)
    return panel, feature_columns


def _extract_feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    """Return a dense, finite, clipped matrix for model fitting."""

    matrix = frame[list(feature_columns)].copy()
    matrix = matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=-1000.0, upper=1000.0)
    return matrix.to_numpy(dtype=float)


def build_time_series_split(
    n_rows: int,
    split_config: SplitConfig,
    horizon_days: int,
) -> TimeSeriesSplit:
    """Create stable train/validation/test segments with an embargo buffer."""

    embargo_rows = max(split_config.embargo_rows, horizon_days)
    min_eval_rows = max(20, horizon_days * 4)
    train_end = max(split_config.min_train_rows, int(n_rows * (1.0 - split_config.validation_ratio - split_config.test_ratio)))
    validation_end = int(n_rows * (1.0 - split_config.test_ratio))
    validation_start = train_end + embargo_rows
    test_start = validation_end + embargo_rows

    if validation_end - validation_start < min_eval_rows or n_rows - test_start < min_eval_rows:
        train_end = max(split_config.min_train_rows, n_rows - (2 * min_eval_rows + 2 * embargo_rows))
        validation_start = train_end + embargo_rows
        validation_end = validation_start + min_eval_rows
        test_start = validation_end + embargo_rows

    if train_end < split_config.min_train_rows:
        raise ValueError(f"Not enough rows to satisfy min_train_rows={split_config.min_train_rows}: n_rows={n_rows}")
    if validation_end <= validation_start:
        raise ValueError(f"Validation split is empty: n_rows={n_rows}")
    if test_start >= n_rows:
        raise ValueError(f"Test split is empty: n_rows={n_rows}")

    return TimeSeriesSplit(
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        n_rows=n_rows,
    )


def _fit_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    regularization_c: float,
    random_seed: int,
) -> Optional[Pipeline]:
    if len(np.unique(y_train)) < 2:
        return None
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=regularization_c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model.fit(X_train, y_train)
    except Exception:
        return None
    return model


def _fit_return_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> Optional[Pipeline]:
    if len(y_train) == 0:
        return None
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model.fit(X_train, y_train)
    except Exception:
        return None
    return model


def _predict_direction_row(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    row: pd.DataFrame,
    regularization_c: float,
    random_seed: int,
) -> Tuple[int, float]:
    y_train = train_frame["target_direction"].astype(int).to_numpy()
    base_rate = float(y_train.mean()) if len(y_train) else 0.5
    if len(y_train) == 0:
        return int(base_rate >= 0.5), base_rate

    model = _fit_direction_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=y_train,
        regularization_c=regularization_c,
        random_seed=random_seed,
    )
    if model is None:
        probability = base_rate
        return int(probability >= 0.5), probability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        probability = float(model.predict_proba(_extract_feature_matrix(row, feature_columns))[0, 1])
    return int(probability >= 0.5), probability


def _predict_return_row(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    row: pd.DataFrame,
    alpha: float,
) -> float:
    y_train = train_frame["target_return"].astype(float).to_numpy()
    if len(y_train) == 0:
        return 0.0

    model = _fit_return_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=y_train,
        alpha=alpha,
    )
    if model is None:
        return float(np.mean(y_train)) if len(y_train) else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(model.predict(_extract_feature_matrix(row, feature_columns))[0])


def walk_forward_direction_predictions(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    evaluation_indices: Sequence[int],
    horizon_days: int,
    regularization_c: float,
    random_seed: int,
    split_label: str,
    refit_every_rows: int,
) -> pd.DataFrame:
    """Generate leakage-safe walk-forward direction predictions."""

    records: List[Dict[str, object]] = []
    cached_model: Optional[Pipeline] = None
    cached_probability = 0.5
    cached_train_rows = 0
    for step_idx, global_idx in enumerate(evaluation_indices):
        history_end = global_idx - horizon_days + 1
        if history_end <= 0:
            continue
        if cached_model is None or step_idx % refit_every_rows == 0:
            train_frame = panel.iloc[:history_end].dropna(subset=["target_direction", "target_return"]).copy()
            if train_frame.empty:
                continue
            cached_train_rows = int(len(train_frame))
            y_train = train_frame["target_direction"].astype(int).to_numpy()
            cached_probability = float(y_train.mean()) if len(y_train) else 0.5
            cached_model = _fit_direction_model(
                X_train=_extract_feature_matrix(train_frame, feature_columns),
                y_train=y_train,
                regularization_c=regularization_c,
                random_seed=random_seed,
            )

        row = panel.iloc[[global_idx]].copy()
        if cached_model is None:
            probability = cached_probability
            prediction = int(probability >= 0.5)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                probability = float(cached_model.predict_proba(_extract_feature_matrix(row, feature_columns))[0, 1])
            prediction = int(probability >= 0.5)
        records.append(
            {
                "Date": row["Date"].iloc[0],
                "Ticker": row["Ticker"].iloc[0],
                "split": split_label,
                "row_index": int(global_idx),
                "history_rows": cached_train_rows,
                "actual_direction": int(row["target_direction"].iloc[0]),
                "actual_return": float(row["target_return"].iloc[0]),
                "predicted_direction": int(prediction),
                "predicted_probability": float(probability),
            }
        )

    return pd.DataFrame(records)


def walk_forward_return_predictions(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    evaluation_indices: Sequence[int],
    horizon_days: int,
    alpha: float,
    split_label: str,
    refit_every_rows: int,
) -> pd.DataFrame:
    """Generate leakage-safe walk-forward return predictions."""

    records: List[Dict[str, object]] = []
    cached_model: Optional[Pipeline] = None
    cached_mean = 0.0
    cached_train_rows = 0
    for step_idx, global_idx in enumerate(evaluation_indices):
        history_end = global_idx - horizon_days + 1
        if history_end <= 0:
            continue
        if cached_model is None or step_idx % refit_every_rows == 0:
            train_frame = panel.iloc[:history_end].dropna(subset=["target_direction", "target_return"]).copy()
            if train_frame.empty:
                continue
            cached_train_rows = int(len(train_frame))
            cached_mean = float(train_frame["target_return"].mean()) if len(train_frame) else 0.0
            cached_model = _fit_return_model(
                X_train=_extract_feature_matrix(train_frame, feature_columns),
                y_train=train_frame["target_return"].astype(float).to_numpy(),
                alpha=alpha,
            )

        row = panel.iloc[[global_idx]].copy()
        if cached_model is None:
            predicted_return = cached_mean
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                predicted_return = float(cached_model.predict(_extract_feature_matrix(row, feature_columns))[0])
        records.append(
            {
                "Date": row["Date"].iloc[0],
                "Ticker": row["Ticker"].iloc[0],
                "split": split_label,
                "row_index": int(global_idx),
                "history_rows": cached_train_rows,
                "actual_direction": int(row["target_direction"].iloc[0]),
                "actual_return": float(row["target_return"].iloc[0]),
                "predicted_return": float(predicted_return),
                "predicted_direction_from_return": int(predicted_return >= 0.0),
            }
        )

    return pd.DataFrame(records)


def fit_in_sample_direction_predictions(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    train_end: int,
    regularization_c: float,
    random_seed: int,
) -> pd.DataFrame:
    """Fit once on the initial train block and save in-sample training predictions."""

    train_frame = panel.iloc[:train_end].dropna(subset=["target_direction", "target_return"]).copy()
    if train_frame.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "split"])

    model = _fit_direction_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=train_frame["target_direction"].astype(int).to_numpy(),
        regularization_c=regularization_c,
        random_seed=random_seed,
    )
    if model is None:
        probability = float(train_frame["target_direction"].mean())
        predictions = np.full(len(train_frame), int(probability >= 0.5))
        probabilities = np.full(len(train_frame), probability)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            probabilities = model.predict_proba(_extract_feature_matrix(train_frame, feature_columns))[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "Date": train_frame["Date"].to_numpy(),
            "Ticker": train_frame["Ticker"].to_numpy(),
            "split": "train",
            "row_index": train_frame.index.to_numpy(),
            "history_rows": len(train_frame),
            "actual_direction": train_frame["target_direction"].astype(int).to_numpy(),
            "actual_return": train_frame["target_return"].astype(float).to_numpy(),
            "predicted_direction": predictions,
            "predicted_probability": probabilities,
        }
    )


def fit_in_sample_return_predictions(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    train_end: int,
    alpha: float,
) -> pd.DataFrame:
    """Fit once on the initial train block and save in-sample training predictions."""

    train_frame = panel.iloc[:train_end].dropna(subset=["target_direction", "target_return"]).copy()
    if train_frame.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "split"])

    model = _fit_return_model(
        X_train=_extract_feature_matrix(train_frame, feature_columns),
        y_train=train_frame["target_return"].astype(float).to_numpy(),
        alpha=alpha,
    )
    if model is None:
        predictions = np.full(len(train_frame), float(train_frame["target_return"].mean()))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            predictions = model.predict(_extract_feature_matrix(train_frame, feature_columns))

    return pd.DataFrame(
        {
            "Date": train_frame["Date"].to_numpy(),
            "Ticker": train_frame["Ticker"].to_numpy(),
            "split": "train",
            "row_index": train_frame.index.to_numpy(),
            "history_rows": len(train_frame),
            "actual_direction": train_frame["target_direction"].astype(int).to_numpy(),
            "actual_return": train_frame["target_return"].astype(float).to_numpy(),
            "predicted_return": predictions,
            "predicted_direction_from_return": (predictions >= 0.0).astype(int),
        }
    )


def compute_direction_metrics(predictions: pd.DataFrame) -> Dict[str, object]:
    """Compute direction metrics for one prediction set."""

    if predictions.empty:
        return {
            "rows": 0,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "balanced_accuracy": math.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_true = predictions["actual_direction"].astype(int).to_numpy()
    y_pred = predictions["predicted_direction"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_return_metrics(predictions: pd.DataFrame) -> Dict[str, object]:
    """Compute regression metrics for one prediction set."""

    if predictions.empty:
        return {
            "rows": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "sign_accuracy": math.nan,
            "return_correlation": math.nan,
        }

    y_true = predictions["actual_return"].astype(float).to_numpy()
    y_pred = predictions["predicted_return"].astype(float).to_numpy()
    sign_accuracy = float((np.sign(y_true) == np.sign(y_pred)).mean())
    if np.std(y_true) <= 0 or np.std(y_pred) <= 0:
        correlation = math.nan
    else:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rows": int(len(predictions)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "sign_accuracy": sign_accuracy,
        "return_correlation": correlation,
    }


def _select_best_direction_hyperparameter(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    split: TimeSeriesSplit,
    config: SentimentOnlyPipelineConfig,
) -> Tuple[float, pd.DataFrame]:
    best_c = config.classification_c_grid[0]
    best_score = (-math.inf, -math.inf, -math.inf)
    best_predictions = pd.DataFrame()

    for regularization_c in config.classification_c_grid:
        predictions = walk_forward_direction_predictions(
            panel=panel,
            feature_columns=feature_columns,
            evaluation_indices=split.validation_indices,
            horizon_days=config.horizon_days,
            regularization_c=regularization_c,
            random_seed=config.random_seed,
            split_label="validation",
            refit_every_rows=config.refit_every_rows,
        )
        metrics = compute_direction_metrics(predictions)
        score = (
            metrics["f1"] if not math.isnan(metrics["f1"]) else -math.inf,
            metrics["balanced_accuracy"] if not math.isnan(metrics["balanced_accuracy"]) else -math.inf,
            metrics["accuracy"] if not math.isnan(metrics["accuracy"]) else -math.inf,
        )
        if score > best_score:
            best_score = score
            best_c = regularization_c
            best_predictions = predictions

    return best_c, best_predictions


def _select_best_return_hyperparameter(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    split: TimeSeriesSplit,
    config: SentimentOnlyPipelineConfig,
) -> Tuple[float, pd.DataFrame]:
    best_alpha = config.regression_alpha_grid[0]
    best_score = (math.inf, math.inf, -math.inf)
    best_predictions = pd.DataFrame()

    for alpha in config.regression_alpha_grid:
        predictions = walk_forward_return_predictions(
            panel=panel,
            feature_columns=feature_columns,
            evaluation_indices=split.validation_indices,
            horizon_days=config.horizon_days,
            alpha=alpha,
            split_label="validation",
            refit_every_rows=config.refit_every_rows,
        )
        metrics = compute_return_metrics(predictions)
        score = (
            metrics["rmse"] if not math.isnan(metrics["rmse"]) else math.inf,
            metrics["mae"] if not math.isnan(metrics["mae"]) else math.inf,
            -(metrics["sign_accuracy"] if not math.isnan(metrics["sign_accuracy"]) else -math.inf),
        )
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_predictions = predictions

    return best_alpha, best_predictions


def _serialize_config(config: SentimentOnlyPipelineConfig) -> Dict[str, object]:
    payload = asdict(config)
    payload["news_data_dir"] = str(config.news_data_dir)
    payload["processed_news_dir"] = str(config.processed_news_dir)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"

    safe_frame = frame.copy().replace([np.inf, -np.inf], math.nan)
    safe_frame = safe_frame.where(pd.notna(safe_frame), "")
    headers = [str(column) for column in safe_frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in safe_frame.iterrows():
        values = [str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


class SentimentOnlyPipeline:
    """End-to-end sentiment-only benchmark for CIB, TMG, and OIH."""

    def __init__(self, config: Optional[SentimentOnlyPipelineConfig] = None):
        self.config = config or SentimentOnlyPipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dirs = {
            "articles": self.output_dir / "articles",
            "prices": self.output_dir / "prices",
            "features": self.output_dir / "features",
            "panels": self.output_dir / "panels",
            "predictions": self.output_dir / "predictions",
            "metrics": self.output_dir / "metrics",
        }
        for path in self.artifact_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, object]:
        raw_news_by_ticker = self._scrape_news()
        price_frames = self._load_price_frames()
        raw_news = self._combine_raw_news(raw_news_by_ticker)

        prepared_articles = prepare_sentiment_articles(
            raw_news=raw_news,
            timezone=self.config.timezone,
            min_match_confidence=self.config.min_match_confidence,
        )
        prepared_path = self.artifact_dirs["articles"] / "prepared_articles.csv"
        prepared_articles.to_csv(prepared_path, index=False)

        scored_articles = score_prepared_articles(prepared_articles=prepared_articles, config=self.config)
        scored_path = self.artifact_dirs["articles"] / "article_sentiment.csv"
        scored_articles.to_csv(scored_path, index=False)

        dataset_sizes = self._build_dataset_size_summary(
            raw_news=raw_news,
            prepared_articles=prepared_articles,
            scored_articles=scored_articles,
            price_frames=price_frames,
        )
        dataset_sizes_path = self.output_dir / "dataset_sizes.csv"
        dataset_sizes.to_csv(dataset_sizes_path, index=False)
        processed_news_paths = self._write_sentiment_ready_outputs(
            raw_news=raw_news,
            prepared_articles=prepared_articles,
            scored_articles=scored_articles,
            dataset_sizes=dataset_sizes,
        )

        direction_metrics_rows: List[Dict[str, object]] = []
        return_metrics_rows: List[Dict[str, object]] = []
        hyperparameter_rows: List[Dict[str, object]] = []
        direction_predictions_all: List[pd.DataFrame] = []
        return_predictions_all: List[pd.DataFrame] = []

        for view in LANGUAGE_VIEWS:
            daily_features = aggregate_daily_sentiment_features(scored_articles=scored_articles, view=view)
            daily_features_path = self.artifact_dirs["features"] / f"daily_sentiment_features_{view}.csv"
            daily_features.to_csv(daily_features_path, index=False)

            for ticker in self.config.tickers:
                price_df = price_frames[ticker]
                ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
                panel, feature_columns = build_sentiment_panel(price_df=price_df, daily_features=ticker_daily, config=self.config)
                panel_path = self.artifact_dirs["panels"] / f"{ticker}_{view}_panel.csv"
                panel.to_csv(panel_path, index=False)

                split = build_time_series_split(
                    n_rows=len(panel),
                    split_config=self.config.split_config,
                    horizon_days=self.config.horizon_days,
                )
                direction_c, val_direction_predictions = _select_best_direction_hyperparameter(
                    panel=panel,
                    feature_columns=feature_columns,
                    split=split,
                    config=self.config,
                )
                return_alpha, val_return_predictions = _select_best_return_hyperparameter(
                    panel=panel,
                    feature_columns=feature_columns,
                    split=split,
                    config=self.config,
                )

                hyperparameter_rows.extend(
                    [
                        {
                            "ticker": ticker,
                            "view": view,
                            "task": "direction",
                            "selected_hyperparameter": direction_c,
                            "hyperparameter_name": "C",
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                        },
                        {
                            "ticker": ticker,
                            "view": view,
                            "task": "return",
                            "selected_hyperparameter": return_alpha,
                            "hyperparameter_name": "alpha",
                            "feature_count": len(feature_columns),
                            "panel_rows": len(panel),
                        },
                    ]
                )

                train_direction_predictions = fit_in_sample_direction_predictions(
                    panel=panel,
                    feature_columns=feature_columns,
                    train_end=split.train_end,
                    regularization_c=direction_c,
                    random_seed=self.config.random_seed,
                )
                test_direction_predictions = walk_forward_direction_predictions(
                    panel=panel,
                    feature_columns=feature_columns,
                    evaluation_indices=split.test_indices,
                    horizon_days=self.config.horizon_days,
                    regularization_c=direction_c,
                    random_seed=self.config.random_seed,
                    split_label="test",
                    refit_every_rows=self.config.refit_every_rows,
                )
                direction_predictions = pd.concat(
                    [train_direction_predictions, val_direction_predictions, test_direction_predictions],
                    ignore_index=True,
                )
                direction_predictions["view"] = view
                direction_predictions["ticker"] = ticker
                direction_predictions["task"] = "direction"
                direction_predictions_all.append(direction_predictions)

                train_return_predictions = fit_in_sample_return_predictions(
                    panel=panel,
                    feature_columns=feature_columns,
                    train_end=split.train_end,
                    alpha=return_alpha,
                )
                test_return_predictions = walk_forward_return_predictions(
                    panel=panel,
                    feature_columns=feature_columns,
                    evaluation_indices=split.test_indices,
                    horizon_days=self.config.horizon_days,
                    alpha=return_alpha,
                    split_label="test",
                    refit_every_rows=self.config.refit_every_rows,
                )
                return_predictions = pd.concat(
                    [train_return_predictions, val_return_predictions, test_return_predictions],
                    ignore_index=True,
                )
                return_predictions["view"] = view
                return_predictions["ticker"] = ticker
                return_predictions["task"] = "return"
                return_predictions_all.append(return_predictions)

                direction_metrics_rows.extend(
                    self._build_direction_metric_rows(
                        ticker=ticker,
                        view=view,
                        predictions=direction_predictions,
                        feature_count=len(feature_columns),
                        panel_rows=len(panel),
                        hyperparameter=direction_c,
                    )
                )
                return_metrics_rows.extend(
                    self._build_return_metric_rows(
                        ticker=ticker,
                        view=view,
                        predictions=return_predictions,
                        feature_count=len(feature_columns),
                        panel_rows=len(panel),
                        hyperparameter=return_alpha,
                    )
                )

        direction_predictions_df = pd.concat(direction_predictions_all, ignore_index=True)
        return_predictions_df = pd.concat(return_predictions_all, ignore_index=True)
        direction_predictions_path = self.artifact_dirs["predictions"] / "direction_predictions.csv"
        return_predictions_path = self.artifact_dirs["predictions"] / "return_predictions.csv"
        direction_predictions_df.to_csv(direction_predictions_path, index=False)
        return_predictions_df.to_csv(return_predictions_path, index=False)

        direction_metrics_df = pd.DataFrame(direction_metrics_rows)
        return_metrics_df = pd.DataFrame(return_metrics_rows)
        direction_metrics_df = pd.concat(
            [direction_metrics_df, self._build_pooled_direction_metrics(direction_predictions_df)],
            ignore_index=True,
        )
        return_metrics_df = pd.concat(
            [return_metrics_df, self._build_pooled_return_metrics(return_predictions_df)],
            ignore_index=True,
        )

        direction_metrics_path = self.artifact_dirs["metrics"] / "direction_metrics.csv"
        return_metrics_path = self.artifact_dirs["metrics"] / "return_metrics.csv"
        hyperparameter_path = self.artifact_dirs["metrics"] / "selected_hyperparameters.csv"
        direction_metrics_df.to_csv(direction_metrics_path, index=False)
        return_metrics_df.to_csv(return_metrics_path, index=False)
        pd.DataFrame(hyperparameter_rows).to_csv(hyperparameter_path, index=False)

        summary = self._build_summary_payload(
            dataset_sizes=dataset_sizes,
            direction_metrics=direction_metrics_df,
            return_metrics=return_metrics_df,
            prepared_articles_path=prepared_path,
            scored_articles_path=scored_path,
            processed_news_paths=processed_news_paths,
            direction_predictions_path=direction_predictions_path,
            return_predictions_path=return_predictions_path,
        )
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary_markdown_path = self.output_dir / "results_summary.md"
        summary_markdown_path.write_text(
            self._render_summary_markdown(dataset_sizes, direction_metrics_df, return_metrics_df),
            encoding="utf-8",
        )

        return {
            "summary": summary,
            "summary_path": summary_path,
            "dataset_sizes_path": dataset_sizes_path,
            "direction_metrics_path": direction_metrics_path,
            "return_metrics_path": return_metrics_path,
            "direction_predictions_path": direction_predictions_path,
            "return_predictions_path": return_predictions_path,
            "results_summary_markdown": summary_markdown_path,
            "output_dir": self.output_dir,
        }

    def _scrape_news(self) -> Dict[str, List[Dict]]:
        scraper_config = AlMobasherScraperConfig.from_env()
        scraper_config = AlMobasherScraperConfig(
            output_dir=self.config.news_data_dir,
            log_dir=scraper_config.log_dir,
            request_timeout_seconds=scraper_config.request_timeout_seconds,
            request_delay_seconds=scraper_config.request_delay_seconds,
            max_retries=scraper_config.max_retries,
            backoff_factor=scraper_config.backoff_factor,
            max_pages_per_language=self.config.scrape_max_pages,
            stop_after_seen_pages=scraper_config.stop_after_seen_pages,
            force_rescrape=self.config.force_scrape,
            include_raw_html=scraper_config.include_raw_html,
            user_agent=scraper_config.user_agent,
            log_level=scraper_config.log_level,
        )
        scraper = AlMobasherScraper(
            config=scraper_config,
            tickers=self.config.tickers,
            languages=self.config.scrape_languages,
        )
        return scraper.run()

    def _load_price_frames(self) -> Dict[str, pd.DataFrame]:
        # Prefer the repo-local cached CSVs. They are reproducible and avoid
        # depending on live network calls during the benchmark run itself.
        local_loader = StockDataLoader(data_dir="data")
        fresh_loader = StockDataLoader(data_dir=str(self.artifact_dirs["prices"]))
        frames: Dict[str, pd.DataFrame] = {}
        for ticker in self.config.tickers:
            try:
                price_df = local_loader.load_ticker(
                    ticker,
                    allow_yfinance_fallback=False,
                    period=self.config.price_period,
                ).copy()
            except Exception:
                price_df = fresh_loader.load_ticker(
                    ticker,
                    allow_yfinance_fallback=True,
                    period=self.config.price_period,
                ).copy()
            price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce").dt.normalize()
            price_df = price_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
            price_df["Ticker"] = ticker
            price_path = self.artifact_dirs["prices"] / f"{ticker}_price_history.csv"
            price_df.to_csv(price_path, index=False)
            frames[ticker] = price_df
        return frames

    def _combine_raw_news(self, raw_news_by_ticker: Dict[str, List[Dict]]) -> pd.DataFrame:
        records: List[Dict] = []
        for ticker in self.config.tickers:
            records.extend(raw_news_by_ticker.get(ticker, []))
        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)

    def _write_jsonl_dataframe(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame.empty:
            path.write_text("", encoding="utf-8")
            return
        frame.to_json(path, orient="records", lines=True, force_ascii=False, date_format="iso")

    def _write_sentiment_ready_outputs(
        self,
        raw_news: pd.DataFrame,
        prepared_articles: pd.DataFrame,
        scored_articles: pd.DataFrame,
        dataset_sizes: pd.DataFrame,
    ) -> Dict[str, str]:
        processed_dir = Path(self.config.processed_news_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_jsonl_path = processed_dir / "raw_scraped_articles.jsonl"
        cleaned_csv_path = processed_dir / "cleaned_articles.csv"
        cleaned_jsonl_path = processed_dir / "cleaned_articles.jsonl"
        sentiment_csv_path = processed_dir / "sentiment_labeled_articles.csv"
        sentiment_jsonl_path = processed_dir / "sentiment_labeled_articles.jsonl"
        summary_path = processed_dir / "dataset_summary_by_ticker_language.csv"
        manifest_path = processed_dir / "manifest.json"

        self._write_jsonl_dataframe(raw_news, raw_jsonl_path)
        prepared_articles.to_csv(cleaned_csv_path, index=False)
        self._write_jsonl_dataframe(prepared_articles, cleaned_jsonl_path)

        ready_columns = [
            "ticker",
            "canonical_ticker",
            "company_name",
            "company_name_en",
            "company_name_ar",
            "source",
            "language",
            "article_language",
            "detected_language",
            "article_id",
            "url",
            "article_url",
            "title",
            "body_text",
            "published_timestamp",
            "scraped_timestamp",
            "matched_aliases",
            "matched_keywords",
            "match_confidence",
            "entity_relevance_label",
            "relevance_flag",
            "relevance_score",
            "duplicate_flag",
            "deduplication_id",
            "near_deduplication_id",
            "cleaned_title",
            "cleaned_body_text",
            "clean_article_text",
            "model_route",
            "chosen_model",
            "sentiment_label",
            "positive_probability",
            "negative_probability",
            "neutral_probability",
            "continuous_sentiment_score",
            "confidence",
            "is_scored",
        ]
        sentiment_ready = scored_articles.copy()
        for column in ready_columns:
            if column not in sentiment_ready.columns:
                sentiment_ready[column] = ""
        sentiment_ready = sentiment_ready[ready_columns].sort_values(
            ["ticker", "published_timestamp", "article_id"],
            na_position="last",
        )
        sentiment_ready.to_csv(sentiment_csv_path, index=False)
        self._write_jsonl_dataframe(sentiment_ready, sentiment_jsonl_path)

        summary_rows: List[Dict[str, object]] = []
        raw_counts = {}
        if not raw_news.empty:
            raw_count_frame = raw_news.copy()
            raw_count_frame["canonical_ticker"] = raw_count_frame.get("ticker", pd.Series(dtype=str)).map(resolve_ticker)
            raw_counts = (
                raw_count_frame.groupby(["canonical_ticker", "article_language"], dropna=False)
                .size()
                .to_dict()
            )
        for ticker in self.config.tickers:
            for language in ("en", "ar", "unknown"):
                prepared_subset = (
                    prepared_articles[
                        (prepared_articles["canonical_ticker"] == ticker)
                        & (prepared_articles["article_language"] == language)
                    ]
                    if not prepared_articles.empty
                    else pd.DataFrame()
                )
                scored_subset = (
                    scored_articles[
                        (scored_articles["canonical_ticker"] == ticker)
                        & (scored_articles["detected_language"].fillna("unknown") == language)
                    ]
                    if not scored_articles.empty
                    else pd.DataFrame()
                )
                raw_count = int(raw_counts.get((ticker, language), 0))
                if raw_count == 0 and prepared_subset.empty and scored_subset.empty:
                    continue
                summary_rows.append(
                    {
                        "ticker": ticker,
                        "language": language,
                        "raw_articles": raw_count,
                        "cleaned_articles": int(len(prepared_subset)),
                        "sentiment_labeled_articles": int((scored_subset.get("is_scored", pd.Series(dtype=int)) == 1).sum())
                        if not scored_subset.empty
                        else 0,
                        "positive_articles": int((scored_subset.get("sentiment_label", pd.Series(dtype=str)) == "positive").sum())
                        if not scored_subset.empty
                        else 0,
                        "negative_articles": int((scored_subset.get("sentiment_label", pd.Series(dtype=str)) == "negative").sum())
                        if not scored_subset.empty
                        else 0,
                        "neutral_articles": int((scored_subset.get("sentiment_label", pd.Series(dtype=str)) == "neutral").sum())
                        if not scored_subset.empty
                        else 0,
                        "first_article_date": str(prepared_subset["Date"].min().date()) if not prepared_subset.empty else "",
                        "last_article_date": str(prepared_subset["Date"].max().date()) if not prepared_subset.empty else "",
                    }
                )

        summary_by_language = pd.DataFrame(summary_rows)
        summary_by_language.to_csv(summary_path, index=False)

        manifest = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "raw_news_dir": str(self.config.news_data_dir),
            "processed_news_dir": str(processed_dir),
            "output_dir": str(self.output_dir),
            "tickers": list(self.config.tickers),
            "languages": list(self.config.scrape_languages),
            "min_match_confidence": self.config.min_match_confidence,
            "rows": {
                "raw_scraped_articles": int(len(raw_news)),
                "cleaned_articles": int(len(prepared_articles)),
                "sentiment_labeled_articles": int((scored_articles.get("is_scored", pd.Series(dtype=int)) == 1).sum())
                if not scored_articles.empty
                else 0,
            },
            "paths": {
                "raw_scraped_articles": str(raw_jsonl_path),
                "cleaned_articles_csv": str(cleaned_csv_path),
                "cleaned_articles_jsonl": str(cleaned_jsonl_path),
                "sentiment_labeled_articles_csv": str(sentiment_csv_path),
                "sentiment_labeled_articles_jsonl": str(sentiment_jsonl_path),
                "dataset_summary_by_ticker_language": str(summary_path),
                "failed_pages": str(Path(self.config.news_data_dir) / "failed_pages.jsonl"),
                "failed_articles": str(Path(self.config.news_data_dir) / "failed_articles.jsonl"),
            },
            "dataset_sizes": dataset_sizes.to_dict(orient="records"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        paths = {key: str(value) for key, value in manifest["paths"].items()}
        paths["manifest"] = str(manifest_path)
        return paths

    def _build_dataset_size_summary(
        self,
        raw_news: pd.DataFrame,
        prepared_articles: pd.DataFrame,
        scored_articles: pd.DataFrame,
        price_frames: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for ticker in self.config.tickers:
            raw_ticker = raw_news[raw_news.get("ticker", pd.Series(dtype=str)).map(resolve_ticker) == ticker] if not raw_news.empty else pd.DataFrame()
            prepared_ticker = prepared_articles[prepared_articles["Ticker"] == ticker] if not prepared_articles.empty else pd.DataFrame()
            scored_ticker = scored_articles[scored_articles["Ticker"] == ticker] if not scored_articles.empty else pd.DataFrame()
            rows.append(
                {
                    "ticker": ticker,
                    "raw_articles": int(len(raw_ticker)),
                    "prepared_articles": int(len(prepared_ticker)),
                    "scored_articles": int((scored_ticker.get("is_scored", pd.Series(dtype=int)) == 1).sum()) if not scored_ticker.empty else 0,
                    "english_articles": int((scored_ticker.get("detected_language", pd.Series(dtype=str)) == "en").sum()) if not scored_ticker.empty else 0,
                    "arabic_articles": int((scored_ticker.get("detected_language", pd.Series(dtype=str)) == "ar").sum()) if not scored_ticker.empty else 0,
                    "price_rows": int(len(price_frames[ticker])),
                    "first_price_date": str(price_frames[ticker]["Date"].min().date()),
                    "last_price_date": str(price_frames[ticker]["Date"].max().date()),
                    "first_article_date": str(prepared_ticker["Date"].min().date()) if not prepared_ticker.empty else "",
                    "last_article_date": str(prepared_ticker["Date"].max().date()) if not prepared_ticker.empty else "",
                }
            )
        return pd.DataFrame(rows)

    def _build_direction_metric_rows(
        self,
        ticker: str,
        view: str,
        predictions: pd.DataFrame,
        feature_count: int,
        panel_rows: int,
        hyperparameter: float,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for split_label in ("validation", "test"):
            split_predictions = predictions[predictions["split"] == split_label].copy()
            metrics = compute_direction_metrics(split_predictions)
            rows.append(
                {
                    "ticker": ticker,
                    "view": view,
                    "split": split_label,
                    "feature_count": feature_count,
                    "panel_rows": panel_rows,
                    "selected_C": hyperparameter,
                    **metrics,
                }
            )
        return rows

    def _build_return_metric_rows(
        self,
        ticker: str,
        view: str,
        predictions: pd.DataFrame,
        feature_count: int,
        panel_rows: int,
        hyperparameter: float,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for split_label in ("validation", "test"):
            split_predictions = predictions[predictions["split"] == split_label].copy()
            metrics = compute_return_metrics(split_predictions)
            rows.append(
                {
                    "ticker": ticker,
                    "view": view,
                    "split": split_label,
                    "feature_count": feature_count,
                    "panel_rows": panel_rows,
                    "selected_alpha": hyperparameter,
                    **metrics,
                }
            )
        return rows

    def _build_pooled_direction_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for view in LANGUAGE_VIEWS:
            for split_label in ("validation", "test"):
                subset = predictions[(predictions["view"] == view) & (predictions["split"] == split_label)].copy()
                metrics = compute_direction_metrics(subset)
                rows.append(
                    {
                        "ticker": "ALL",
                        "view": view,
                        "split": split_label,
                        "feature_count": math.nan,
                        "panel_rows": math.nan,
                        "selected_C": math.nan,
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    def _build_pooled_return_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for view in LANGUAGE_VIEWS:
            for split_label in ("validation", "test"):
                subset = predictions[(predictions["view"] == view) & (predictions["split"] == split_label)].copy()
                metrics = compute_return_metrics(subset)
                rows.append(
                    {
                        "ticker": "ALL",
                        "view": view,
                        "split": split_label,
                        "feature_count": math.nan,
                        "panel_rows": math.nan,
                        "selected_alpha": math.nan,
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    def _build_summary_payload(
        self,
        dataset_sizes: pd.DataFrame,
        direction_metrics: pd.DataFrame,
        return_metrics: pd.DataFrame,
        prepared_articles_path: Path,
        scored_articles_path: Path,
        processed_news_paths: Dict[str, str],
        direction_predictions_path: Path,
        return_predictions_path: Path,
    ) -> Dict[str, object]:
        return {
            "config": _serialize_config(self.config),
            "dataset_sizes": dataset_sizes.to_dict(orient="records"),
            "direction_test_metrics": direction_metrics[direction_metrics["split"] == "test"].to_dict(orient="records"),
            "return_test_metrics": return_metrics[return_metrics["split"] == "test"].to_dict(orient="records"),
            "artifacts": {
                "prepared_articles": str(prepared_articles_path),
                "scored_articles": str(scored_articles_path),
                "processed_news": processed_news_paths,
                "direction_predictions": str(direction_predictions_path),
                "return_predictions": str(return_predictions_path),
            },
        }

    def _render_summary_markdown(
        self,
        dataset_sizes: pd.DataFrame,
        direction_metrics: pd.DataFrame,
        return_metrics: pd.DataFrame,
    ) -> str:
        lines = [
            "# Sentiment-Only Pipeline Results",
            "",
            "## Dataset Sizes",
            "",
            _dataframe_to_markdown(dataset_sizes),
            "",
            "## Direction Test Metrics",
            "",
            _dataframe_to_markdown(direction_metrics[direction_metrics["split"] == "test"]),
            "",
            "## Return Test Metrics",
            "",
            _dataframe_to_markdown(return_metrics[return_metrics["split"] == "test"]),
            "",
        ]
        return "\n".join(lines)
