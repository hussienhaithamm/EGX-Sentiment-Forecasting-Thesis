"""Daily feature aggregation and price-row alignment for Mubasher news."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .loader import NewsDataLoader
from .tickers import TARGET_COMPANIES, resolve_ticker

_CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

_BASE_FEATURE_COLUMNS = [
    "news_count_total",
    "news_count_ar",
    "news_count_en",
    "news_has_news",
    "news_unique_urls",
    "news_title_len_mean",
    "news_body_len_mean",
    "news_high_conf_count",
    "news_medium_conf_count",
    "news_low_conf_count",
]

_COUNT_FEATURE_COLUMNS = [
    "news_count_total",
    "news_count_ar",
    "news_count_en",
    "news_has_news",
    "news_unique_urls",
    "news_high_conf_count",
    "news_medium_conf_count",
    "news_low_conf_count",
]

_ROLLING_COUNT_COLUMNS = [
    "news_count_total",
    "news_count_ar",
    "news_count_en",
    "news_high_conf_count",
    "news_medium_conf_count",
    "news_low_conf_count",
]

_FLOAT_FEATURE_COLUMNS = [
    "news_title_len_mean",
    "news_body_len_mean",
]

_EMPTY_DAILY_COLUMNS = ["Ticker", "Date"] + _BASE_FEATURE_COLUMNS
_INTERNAL_TICKER_COLUMN = "__news_ticker"
_INTERNAL_DATE_COLUMN = "__news_merge_date"


@dataclass(frozen=True)
class NewsFeatureConfig:
    """Configuration for daily news aggregation and safe feature alignment."""

    news_data_dir: Path = Path("data/raw/news/al_mobasher")
    timezone: str = "Africa/Cairo"
    date_assignment: str = "local_calendar_date"
    lag_strategy: str = "previous_day"
    rolling_windows: Tuple[int, ...] = (3, 7)
    min_match_confidence: str = "low"

    def __post_init__(self) -> None:
        if self.date_assignment != "local_calendar_date":
            raise ValueError("date_assignment must be 'local_calendar_date'")
        if self.lag_strategy not in {"previous_day", "same_day_close_safe"}:
            raise ValueError("lag_strategy must be 'previous_day' or 'same_day_close_safe'")
        if self.min_match_confidence not in _CONFIDENCE_RANK:
            raise ValueError("min_match_confidence must be one of: low, medium, high")
        if not self.rolling_windows:
            raise ValueError("rolling_windows must contain at least one window")
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling_windows must contain only positive integers")


class MubasherNewsFeatureBuilder:
    """Turn article-level Mubasher data into daily, leak-safe price features."""

    def __init__(
        self,
        config: Optional[NewsFeatureConfig] = None,
        loader: Optional[NewsDataLoader] = None,
    ):
        self.config = config or NewsFeatureConfig()
        self.loader = loader or NewsDataLoader(str(self.config.news_data_dir))
        self.logger = logging.getLogger(__name__)

    @property
    def base_feature_columns(self) -> List[str]:
        """Return base one-day aggregated feature columns."""

        return list(_BASE_FEATURE_COLUMNS)

    @property
    def feature_columns(self) -> List[str]:
        """Return all aligned feature columns produced for price rows."""

        columns = list(_BASE_FEATURE_COLUMNS)
        for window in self.config.rolling_windows:
            for feature_name in _ROLLING_COUNT_COLUMNS:
                columns.append(f"{feature_name}_roll{window}d")
        return columns

    def load_raw_news(self, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Load raw ticker-level JSONL files into a single dataframe."""

        if tickers is None:
            tickers = sorted(TARGET_COMPANIES.keys())
        canonical_tickers = [resolve_ticker(ticker) for ticker in tickers]
        canonical_tickers = [ticker for ticker in canonical_tickers if ticker]
        self.loader.load_all(canonical_tickers)

        frames: List[pd.DataFrame] = []
        for ticker in canonical_tickers:
            frame = self.loader.get_raw(ticker)
            if frame.empty:
                continue
            frames.append(frame)

        if not frames:
            self.logger.info("No raw news records found for tickers: %s", canonical_tickers)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.logger.info("Loaded %s raw news records for %s tickers", len(combined), len(frames))
        return combined

    def aggregate_daily(self, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Aggregate article-level records into one row per ticker and local date."""

        raw_news = self.load_raw_news(tickers=tickers)
        if raw_news.empty:
            return pd.DataFrame(columns=_EMPTY_DAILY_COLUMNS)

        prepared = raw_news.copy()
        prepared["Ticker"] = prepared["ticker"].map(resolve_ticker)
        prepared["published_at"] = pd.to_datetime(prepared["published_at"], utc=True, errors="coerce")
        if "match_confidence" in prepared.columns:
            prepared["match_confidence"] = prepared["match_confidence"].fillna("low").astype(str).str.lower()
        else:
            prepared["match_confidence"] = "low"
        prepared = prepared.dropna(subset=["Ticker", "published_at"])

        minimum_rank = _CONFIDENCE_RANK[self.config.min_match_confidence]
        prepared = prepared[
            prepared["match_confidence"].map(lambda value: _CONFIDENCE_RANK.get(value, -1)) >= minimum_rank
        ].copy()

        if prepared.empty:
            self.logger.info("No news records remain after confidence filtering.")
            return pd.DataFrame(columns=_EMPTY_DAILY_COLUMNS)

        published_local = prepared["published_at"].dt.tz_convert(self.config.timezone)
        prepared["Date"] = published_local.dt.tz_localize(None).dt.normalize()
        prepared["article_title_len"] = prepared["article_title"].fillna("").astype(str).str.len()
        prepared["article_body_len"] = prepared["article_body"].fillna("").astype(str).str.len()

        grouped = prepared.groupby(["Ticker", "Date"], as_index=False).agg(
            news_count_total=("article_id", "size"),
            news_count_ar=("article_language", lambda values: int((values == "ar").sum())),
            news_count_en=("article_language", lambda values: int((values == "en").sum())),
            news_unique_urls=("article_url", pd.Series.nunique),
            news_title_len_mean=("article_title_len", "mean"),
            news_body_len_mean=("article_body_len", "mean"),
            news_high_conf_count=("match_confidence", lambda values: int((values == "high").sum())),
            news_medium_conf_count=("match_confidence", lambda values: int((values == "medium").sum())),
            news_low_conf_count=("match_confidence", lambda values: int((values == "low").sum())),
        )
        grouped["news_has_news"] = (grouped["news_count_total"] > 0).astype(int)

        for column in _COUNT_FEATURE_COLUMNS:
            grouped[column] = grouped[column].fillna(0).astype(int)
        for column in _FLOAT_FEATURE_COLUMNS:
            grouped[column] = grouped[column].fillna(0.0).astype(float)

        ordered_columns = ["Ticker", "Date"] + _BASE_FEATURE_COLUMNS
        grouped = grouped[ordered_columns].sort_values(["Ticker", "Date"]).reset_index(drop=True)
        self.logger.info("Aggregated raw news into %s daily ticker rows", len(grouped))
        return grouped

    def merge_with_price_data(
        self,
        price_df: pd.DataFrame,
        daily_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge leak-safe daily news features onto price rows."""

        if price_df.empty:
            return price_df.copy()
        if "Date" not in price_df.columns or "Ticker" not in price_df.columns:
            self.logger.warning("News features require both 'Date' and 'Ticker' columns. Skipping merge.")
            return price_df.copy()

        price = price_df.copy()
        price[_INTERNAL_TICKER_COLUMN] = price["Ticker"].map(resolve_ticker)
        price[_INTERNAL_DATE_COLUMN] = self._normalize_price_dates(price["Date"])

        candidate_tickers = [
            ticker
            for ticker in sorted(price[_INTERNAL_TICKER_COLUMN].dropna().unique())
            if ticker in TARGET_COMPANIES
        ]
        if daily_features is None:
            daily_features = self.aggregate_daily(candidate_tickers)

        unique_pairs = price[[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN]].dropna().drop_duplicates()
        feature_frames: List[pd.DataFrame] = []
        for ticker, ticker_pairs in unique_pairs.groupby(_INTERNAL_TICKER_COLUMN):
            ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
            feature_frames.append(self._align_features_for_ticker(ticker, ticker_pairs, ticker_daily))

        if feature_frames:
            aligned_features = pd.concat(feature_frames, ignore_index=True)
        else:
            aligned_features = pd.DataFrame(columns=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN] + self.feature_columns)

        merged = price.merge(
            aligned_features,
            how="left",
            on=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN],
        )
        for column in self.feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0

        count_columns = [column for column in self.feature_columns if column not in _FLOAT_FEATURE_COLUMNS]
        merged[count_columns] = merged[count_columns].fillna(0).astype(int)
        merged[_FLOAT_FEATURE_COLUMNS] = merged[_FLOAT_FEATURE_COLUMNS].fillna(0.0).astype(float)

        matched_rows = (merged["news_count_total"] > 0).groupby(merged["Ticker"]).sum().to_dict()
        self.logger.info(
            "Merged news features onto %s price rows. Matched rows with prior news: %s",
            len(merged),
            matched_rows,
        )
        return merged.drop(columns=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN])

    def _align_features_for_ticker(
        self,
        ticker: str,
        ticker_pairs: pd.DataFrame,
        daily_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align daily aggregated news to the available trading dates for one ticker."""

        price_dates = pd.Index(sorted(ticker_pairs[_INTERNAL_DATE_COLUMN].dropna().unique()))
        if price_dates.empty:
            return pd.DataFrame(columns=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN] + self.feature_columns)

        if daily_features.empty:
            base_daily = pd.DataFrame(index=pd.date_range(start=price_dates.min(), end=price_dates.max(), freq="D"))
            for column in _BASE_FEATURE_COLUMNS:
                base_daily[column] = 0.0
        else:
            start_date = min(price_dates.min(), daily_features["Date"].min())
            end_date = price_dates.max()
            calendar_index = pd.date_range(start=start_date, end=end_date, freq="D")
            indexed = daily_features.set_index("Date")[_BASE_FEATURE_COLUMNS].reindex(calendar_index)
            for column in _COUNT_FEATURE_COLUMNS:
                indexed[column] = indexed[column].fillna(0).astype(int)
            for column in _FLOAT_FEATURE_COLUMNS:
                indexed[column] = indexed[column].fillna(0.0).astype(float)
            base_daily = indexed

        lag_steps = 1 if self.config.lag_strategy == "previous_day" else 0
        aligned = base_daily[_BASE_FEATURE_COLUMNS].shift(lag_steps)
        for column in _COUNT_FEATURE_COLUMNS:
            aligned[column] = aligned[column].fillna(0).astype(int)
        for column in _FLOAT_FEATURE_COLUMNS:
            aligned[column] = aligned[column].fillna(0.0).astype(float)

        for window in self.config.rolling_windows:
            for column in _ROLLING_COUNT_COLUMNS:
                rolling_values = (
                    base_daily[column]
                    .shift(lag_steps)
                    .rolling(window=window, min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype(int)
                )
                aligned[f"{column}_roll{window}d"] = rolling_values

        aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": _INTERNAL_DATE_COLUMN})
        aligned[_INTERNAL_TICKER_COLUMN] = ticker
        return aligned[[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN] + self.feature_columns]

    def _normalize_price_dates(self, series: pd.Series) -> pd.Series:
        """Normalize price dates to local naive dates for feature merging."""

        dates = pd.to_datetime(series, errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_convert(self.config.timezone).dt.tz_localize(None)
        return dates.dt.normalize()
