"""Bilingual article scoring and sentiment-aware daily Mubasher features."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .features import (
    _CONFIDENCE_RANK,
    _COUNT_FEATURE_COLUMNS,
    _FLOAT_FEATURE_COLUMNS,
    _INTERNAL_DATE_COLUMN,
    _INTERNAL_TICKER_COLUMN,
    _ROLLING_COUNT_COLUMNS,
    MubasherNewsFeatureBuilder,
)
from .tickers import resolve_ticker


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

_SENTIMENT_COUNT_COLUMNS = [
    "sentiment_article_count_total",
    "sentiment_article_count_en",
    "sentiment_article_count_ar",
    "sentiment_scored_count_total",
    "sentiment_positive_count",
    "sentiment_negative_count",
    "sentiment_neutral_count",
]

_SENTIMENT_FLOAT_COLUMNS = [
    "sentiment_score_mean",
    "sentiment_score_mean_en",
    "sentiment_score_mean_ar",
    "sentiment_confidence_mean",
]

_SENTIMENT_ROLLING_SUM_COLUMNS = list(_SENTIMENT_COUNT_COLUMNS)
_SENTIMENT_ROLLING_MEAN_COLUMNS = list(_SENTIMENT_FLOAT_COLUMNS)

_ARTICLE_SENTIMENT_COLUMNS = [
    "ticker",
    "article_id",
    "published_at",
    "date",
    "article_url",
    "source_language",
    "detected_language",
    "language_routing",
    "chosen_model",
    "raw_model_label",
    "sentiment_label",
    "confidence",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "mixed_prob",
    "sentiment_score",
    "is_scored",
]


@dataclass(frozen=True)
class FinBERTConfig:
    """Shared config for the bilingual English/Arabic article sentiment pass."""

    english_model_name: str = "ProsusAI/finbert"
    arabic_model_name: str = "PRAli22/AraBert-Arabic-Sentiment-Analysis"
    batch_size: int = 16
    max_length: int = 256
    article_output_path: Optional[Path] = None


class FinBERTSentimentNewsFeatureBuilder(MubasherNewsFeatureBuilder):
    """Route English news to FinBERT and Arabic news to AraBERT, then aggregate safely."""

    def __init__(self, *args, finbert_config: Optional[FinBERTConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.finbert_config = finbert_config or FinBERTConfig()
        self._resources: Dict[str, Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}
        self._article_sentiment_cache: Optional[pd.DataFrame] = None

    @property
    def sentiment_base_columns(self) -> List[str]:
        return list(_SENTIMENT_COUNT_COLUMNS) + list(_SENTIMENT_FLOAT_COLUMNS)

    @property
    def base_feature_columns(self) -> List[str]:
        return super().base_feature_columns + self.sentiment_base_columns

    @property
    def feature_columns(self) -> List[str]:
        columns = list(super().feature_columns) + self.sentiment_base_columns
        for window in self.config.rolling_windows:
            for column in _SENTIMENT_ROLLING_SUM_COLUMNS:
                columns.append(f"{column}_roll{window}d")
            for column in _SENTIMENT_ROLLING_MEAN_COLUMNS:
                columns.append(f"{column}_roll{window}d")
        return columns

    @property
    def float_feature_columns(self) -> List[str]:
        columns = list(_FLOAT_FEATURE_COLUMNS) + list(_SENTIMENT_FLOAT_COLUMNS)
        for window in self.config.rolling_windows:
            for column in _SENTIMENT_ROLLING_MEAN_COLUMNS:
                columns.append(f"{column}_roll{window}d")
        return columns

    @property
    def count_feature_columns(self) -> List[str]:
        float_columns = set(self.float_feature_columns)
        return [column for column in self.feature_columns if column not in float_columns]

    @property
    def count_feature_columns_base(self) -> List[str]:
        return list(_COUNT_FEATURE_COLUMNS) + list(_SENTIMENT_COUNT_COLUMNS)

    @property
    def float_feature_columns_base(self) -> List[str]:
        return list(_FLOAT_FEATURE_COLUMNS) + list(_SENTIMENT_FLOAT_COLUMNS)

    def aggregate_daily(self, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        raw_news = self.load_raw_news(tickers=tickers)
        if raw_news.empty:
            return pd.DataFrame(columns=["Ticker", "Date"] + self.base_feature_columns)

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
            return pd.DataFrame(columns=["Ticker", "Date"] + self.base_feature_columns)

        published_local = prepared["published_at"].dt.tz_convert(self.config.timezone)
        prepared["Date"] = published_local.dt.tz_localize(None).dt.normalize()
        prepared["article_title_len"] = prepared["article_title"].fillna("").astype(str).str.len()
        prepared["article_body_len"] = prepared["article_body"].fillna("").astype(str).str.len()

        base_grouped = prepared.groupby(["Ticker", "Date"], as_index=False).agg(
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
        base_grouped["news_has_news"] = (base_grouped["news_count_total"] > 0).astype(int)

        article_sentiment = self.score_articles(prepared)
        sentiment_grouped = article_sentiment.groupby(["Ticker", "Date"]).apply(
            self._summarize_daily_sentiment,
            include_groups=False,
        ).reset_index()

        grouped = base_grouped.merge(sentiment_grouped, how="left", on=["Ticker", "Date"])
        for column in self.count_feature_columns_base:
            grouped[column] = grouped[column].fillna(0).astype(int)
        for column in self.float_feature_columns_base:
            grouped[column] = grouped[column].fillna(0.0).astype(float)

        ordered_columns = ["Ticker", "Date"] + self.base_feature_columns
        return grouped[ordered_columns].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def score_articles(self, prepared_news: pd.DataFrame) -> pd.DataFrame:
        """Score English with FinBERT and Arabic with AraBERT, then export article artifacts."""

        if self._article_sentiment_cache is not None:
            return self._article_sentiment_cache.copy()

        scored = prepared_news.copy()
        scored["article_text"] = scored.apply(self._compose_article_text, axis=1)
        scored["source_language"] = scored["article_language"].apply(self._normalize_language)

        language_details = scored.apply(self._resolve_language, axis=1)
        scored["detected_language"] = language_details.apply(lambda item: item[0])
        scored["language_routing"] = language_details.apply(lambda item: item[1])
        scored["chosen_model"] = ""
        scored["raw_model_label"] = "unscored"
        scored["sentiment_label"] = "unscored"
        scored["confidence"] = 0.0
        scored["positive_prob"] = 0.0
        scored["negative_prob"] = 0.0
        scored["neutral_prob"] = 0.0
        scored["mixed_prob"] = 0.0
        scored["sentiment_score"] = 0.0
        scored["is_scored"] = 0
        scored["date"] = scored["Date"]

        source_counts = scored["source_language"].value_counts(dropna=False).to_dict()
        routing_counts = scored["detected_language"].value_counts(dropna=False).to_dict()
        routing_reasons = scored["language_routing"].value_counts(dropna=False).to_dict()
        self.logger.info(
            "Bilingual sentiment routing source=%s detected=%s reasons=%s",
            source_counts,
            routing_counts,
            routing_reasons,
        )

        for language_code, model_name in (
            ("en", self.finbert_config.english_model_name),
            ("ar", self.finbert_config.arabic_model_name),
        ):
            mask = (scored["detected_language"] == language_code) & scored["article_text"].str.len().fillna(0).gt(0)
            if not mask.any():
                continue
            language_rows = scored.loc[mask, ["article_text"]].copy()
            probabilities = self._score_texts(language_rows["article_text"].tolist(), model_name=model_name)
            normalized = probabilities.apply(self._normalize_probabilities, axis=1)
            normalized_df = pd.DataFrame(list(normalized), index=language_rows.index)
            scored.loc[language_rows.index, "chosen_model"] = model_name
            for column in normalized_df.columns:
                scored.loc[language_rows.index, column] = normalized_df[column]
            scored.loc[language_rows.index, "is_scored"] = 1

        if self.finbert_config.article_output_path is not None:
            output_path = Path(self.finbert_config.article_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_columns = [column for column in _ARTICLE_SENTIMENT_COLUMNS if column in scored.columns]
            scored[export_columns].to_csv(output_path, index=False)

        scored_count = int(scored["is_scored"].sum())
        by_model = scored.loc[scored["is_scored"] == 1, "chosen_model"].value_counts().to_dict()
        self.logger.info("Scored %s articles using bilingual routing: %s", scored_count, by_model)

        self._article_sentiment_cache = scored
        return scored.copy()

    def merge_with_price_data(
        self,
        price_df: pd.DataFrame,
        daily_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if price_df.empty:
            return price_df.copy()
        if "Date" not in price_df.columns or "Ticker" not in price_df.columns:
            self.logger.warning("News features require both 'Date' and 'Ticker' columns. Skipping merge.")
            return price_df.copy()

        price = price_df.copy()
        price[_INTERNAL_TICKER_COLUMN] = price["Ticker"].map(resolve_ticker)
        price[_INTERNAL_DATE_COLUMN] = self._normalize_price_dates(price["Date"])

        candidate_tickers = sorted(price[_INTERNAL_TICKER_COLUMN].dropna().unique())
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

        merged = price.merge(aligned_features, how="left", on=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN])
        for column in self.count_feature_columns:
            if column not in merged.columns:
                merged[column] = 0
        for column in self.float_feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0

        merged[self.count_feature_columns] = merged[self.count_feature_columns].fillna(0).astype(int)
        merged[self.float_feature_columns] = merged[self.float_feature_columns].fillna(0.0).astype(float)
        return merged.drop(columns=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN])

    def _align_features_for_ticker(
        self,
        ticker: str,
        ticker_pairs: pd.DataFrame,
        daily_features: pd.DataFrame,
    ) -> pd.DataFrame:
        price_dates = pd.Index(sorted(ticker_pairs[_INTERNAL_DATE_COLUMN].dropna().unique()))
        if price_dates.empty:
            return pd.DataFrame(columns=[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN] + self.feature_columns)

        if daily_features.empty:
            base_daily = pd.DataFrame(index=pd.date_range(start=price_dates.min(), end=price_dates.max(), freq="D"))
            for column in self.base_feature_columns:
                base_daily[column] = 0.0
        else:
            start_date = min(price_dates.min(), daily_features["Date"].min())
            end_date = price_dates.max()
            calendar_index = pd.date_range(start=start_date, end=end_date, freq="D")
            base_daily = daily_features.set_index("Date")[self.base_feature_columns].reindex(calendar_index)
            base_daily[self.count_feature_columns_base] = base_daily[self.count_feature_columns_base].fillna(0).astype(int)
            base_daily[self.float_feature_columns_base] = base_daily[self.float_feature_columns_base].fillna(0.0).astype(float)

        lag_steps = 1 if self.config.lag_strategy == "previous_day" else 0
        aligned = base_daily[self.base_feature_columns].shift(lag_steps)
        aligned[self.count_feature_columns_base] = aligned[self.count_feature_columns_base].fillna(0).astype(int)
        aligned[self.float_feature_columns_base] = aligned[self.float_feature_columns_base].fillna(0.0).astype(float)

        for window in self.config.rolling_windows:
            for column in _ROLLING_COUNT_COLUMNS:
                aligned[f"{column}_roll{window}d"] = (
                    base_daily[column].shift(lag_steps).rolling(window=window, min_periods=1).sum().fillna(0).astype(int)
                )
            for column in _SENTIMENT_ROLLING_SUM_COLUMNS:
                aligned[f"{column}_roll{window}d"] = (
                    base_daily[column].shift(lag_steps).rolling(window=window, min_periods=1).sum().fillna(0).astype(int)
                )
            for column in _SENTIMENT_ROLLING_MEAN_COLUMNS:
                aligned[f"{column}_roll{window}d"] = (
                    base_daily[column].shift(lag_steps).rolling(window=window, min_periods=1).mean().fillna(0.0).astype(float)
                )

        aligned = aligned.loc[price_dates].reset_index().rename(columns={"index": _INTERNAL_DATE_COLUMN})
        aligned[_INTERNAL_TICKER_COLUMN] = ticker
        return aligned[[_INTERNAL_TICKER_COLUMN, _INTERNAL_DATE_COLUMN] + self.feature_columns]

    def _summarize_daily_sentiment(self, group: pd.DataFrame) -> pd.Series:
        scored = group[group["is_scored"] == 1].copy()
        if scored.empty:
            return pd.Series(
                {
                    "sentiment_article_count_total": int(len(group)),
                    "sentiment_article_count_en": int((group["detected_language"] == "en").sum()),
                    "sentiment_article_count_ar": int((group["detected_language"] == "ar").sum()),
                    "sentiment_scored_count_total": 0,
                    "sentiment_score_mean": 0.0,
                    "sentiment_score_mean_en": 0.0,
                    "sentiment_score_mean_ar": 0.0,
                    "sentiment_positive_count": 0,
                    "sentiment_negative_count": 0,
                    "sentiment_neutral_count": 0,
                    "sentiment_confidence_mean": 0.0,
                }
            )

        scored_en = scored[scored["detected_language"] == "en"]
        scored_ar = scored[scored["detected_language"] == "ar"]
        return pd.Series(
            {
                "sentiment_article_count_total": int(len(group)),
                "sentiment_article_count_en": int((group["detected_language"] == "en").sum()),
                "sentiment_article_count_ar": int((group["detected_language"] == "ar").sum()),
                "sentiment_scored_count_total": int(len(scored)),
                "sentiment_score_mean": float(scored["sentiment_score"].mean()),
                "sentiment_score_mean_en": float(scored_en["sentiment_score"].mean()) if not scored_en.empty else 0.0,
                "sentiment_score_mean_ar": float(scored_ar["sentiment_score"].mean()) if not scored_ar.empty else 0.0,
                "sentiment_positive_count": int((scored["sentiment_label"] == "positive").sum()),
                "sentiment_negative_count": int((scored["sentiment_label"] == "negative").sum()),
                "sentiment_neutral_count": int((scored["sentiment_label"] == "neutral").sum()),
                "sentiment_confidence_mean": float(scored["confidence"].mean()),
            }
        )

    def _compose_article_text(self, row: pd.Series) -> str:
        parts = []
        for column in ("article_title", "article_summary", "article_body"):
            value = row.get(column, "")
            text = str(value).strip() if pd.notna(value) else ""
            if text and text not in parts:
                parts.append(text)
        return "\n\n".join(parts)

    def _normalize_language(self, value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        normalized = str(value).strip().lower()
        mapping = {
            "en": "en",
            "english": "en",
            "ar": "ar",
            "arabic": "ar",
        }
        return mapping.get(normalized, "")

    def _resolve_language(self, row: pd.Series) -> Tuple[str, str]:
        source_language = str(row.get("source_language", "") or "")
        guess, arabic_count, latin_count = self._guess_language_from_text(row.get("article_text", ""))

        if source_language not in {"en", "ar"}:
            if guess == "unknown":
                return "unknown", "undetermined"
            return guess, "unicode_fallback"

        if source_language == "en" and guess == "ar" and arabic_count >= max(5, latin_count * 2):
            return "ar", "unicode_override"
        if source_language == "ar" and guess == "en" and latin_count >= max(12, arabic_count * 3):
            return "en", "unicode_override"
        return source_language, "source_language"

    def _score_texts(self, texts: List[str], model_name: str) -> pd.DataFrame:
        tokenizer, model, device = self._load_model(model_name)
        rows: List[Dict[str, object]] = []
        for start in range(0, len(texts), self.finbert_config.batch_size):
            batch_texts = texts[start : start + self.finbert_config.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.finbert_config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = {int(idx): str(label).lower() for idx, label in model.config.id2label.items()}
            for prob_row in probs:
                row = {labels[idx]: float(prob_row[idx]) for idx in range(len(prob_row))}
                rows.append(row)
        return pd.DataFrame(rows)

    def _normalize_probabilities(self, row: pd.Series) -> Dict[str, object]:
        positive_prob = float(row.get("positive", 0.0))
        negative_prob = float(row.get("negative", 0.0))
        mixed_prob = float(row.get("mixed", 0.0))
        neutral_prob = float(row.get("neutral", 0.0)) + mixed_prob
        total = positive_prob + negative_prob + neutral_prob
        if total <= 0:
            total = 1.0
        positive_prob /= total
        negative_prob /= total
        neutral_prob /= total

        shared_probs = {
            "positive": positive_prob,
            "negative": negative_prob,
            "neutral": neutral_prob,
        }
        sentiment_label = max(shared_probs, key=shared_probs.get)
        confidence = float(shared_probs[sentiment_label])

        raw_label_probs = {str(key).lower(): float(value) for key, value in row.to_dict().items()}
        raw_model_label = max(raw_label_probs, key=raw_label_probs.get)

        return {
            "raw_model_label": raw_model_label,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "neutral_prob": neutral_prob,
            "mixed_prob": mixed_prob,
            "sentiment_score": positive_prob - negative_prob,
        }

    def _load_model(
        self,
        model_name: str,
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
        if model_name in self._resources:
            return self._resources[model_name]

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._resources[model_name] = (tokenizer, model, device)
        return self._resources[model_name]

    def _guess_language_from_text(self, text: object) -> Tuple[str, int, int]:
        content = str(text or "")
        arabic_count = len(_ARABIC_CHAR_RE.findall(content))
        latin_count = len(_LATIN_CHAR_RE.findall(content))
        if arabic_count == 0 and latin_count == 0:
            return "unknown", arabic_count, latin_count
        if arabic_count >= max(3, latin_count):
            return "ar", arabic_count, latin_count
        return "en", arabic_count, latin_count
