"""Rebuild sentiment_labeled_articles.csv from existing raw JSONL without scraping."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from news_pipeline.storage import load_jsonl_records
from egx_direction.sentiment_only_pipeline import (
    SentimentOnlyPipelineConfig,
    prepare_sentiment_articles,
    score_prepared_articles,
)

TICKERS = ["CIB", "TMG", "OIH"]
RAW_DIR = REPO_ROOT / "data/raw/news/mubasher_sentiment_only"
PROCESSED_DIR = REPO_ROOT / "data/processed/news/sentiment_ready"

READY_COLUMNS = [
    "ticker", "canonical_ticker", "company_name", "company_name_en", "company_name_ar",
    "source", "language", "article_language", "detected_language", "article_id", "url",
    "article_url", "title", "body_text", "published_timestamp", "scraped_timestamp",
    "matched_aliases", "matched_keywords", "match_confidence", "entity_relevance_label",
    "relevance_flag", "relevance_score", "duplicate_flag", "deduplication_id",
    "near_deduplication_id", "cleaned_title", "cleaned_body_text", "clean_article_text",
    "model_route", "chosen_model", "sentiment_label", "positive_probability",
    "negative_probability", "neutral_probability", "continuous_sentiment_score",
    "confidence", "is_scored",
]


def main() -> None:
    records = []
    for ticker in TICKERS:
        path = RAW_DIR / f"{ticker}.jsonl"
        ticker_records = load_jsonl_records(path)
        print(f"  Loaded {len(ticker_records)} raw records for {ticker}")
        records.extend(ticker_records)

    print(f"Total raw records: {len(records)}")
    raw_news = pd.DataFrame.from_records(records)

    config = SentimentOnlyPipelineConfig()

    print("Preparing articles (filtering, cleaning, deduplication)...")
    prepared = prepare_sentiment_articles(
        raw_news=raw_news,
        timezone=config.timezone,
        min_match_confidence=config.min_match_confidence,
    )
    print(f"Prepared: {len(prepared)} articles")
    print("  By ticker:", prepared["canonical_ticker"].value_counts().to_dict() if "canonical_ticker" in prepared.columns else "N/A")

    print("Scoring articles (FinBERT / AraBERT)...")
    scored = score_prepared_articles(prepared_articles=prepared, config=config)
    print(f"Scored: {len(scored)} articles, {scored['is_scored'].sum()} with labels")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "sentiment_labeled_articles.csv"

    sentiment_ready = scored.copy()
    for col in READY_COLUMNS:
        if col not in sentiment_ready.columns:
            sentiment_ready[col] = ""
    sentiment_ready = (
        sentiment_ready[READY_COLUMNS]
        .sort_values(["ticker", "published_timestamp", "article_id"], na_position="last")
    )
    sentiment_ready.to_csv(out_path, index=False)
    print(f"Written: {out_path} ({len(sentiment_ready)} rows)")


if __name__ == "__main__":
    main()
