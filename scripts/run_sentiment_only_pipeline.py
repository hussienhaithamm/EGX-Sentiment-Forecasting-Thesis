"""Run the generalized sentiment-only EGX pipeline for CIB, TMG, and OIH."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.sentiment_only_pipeline import SentimentOnlyPipeline, SentimentOnlyPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EGX sentiment-only direction/return pipeline.")
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"], help="Target EGX tickers.")
    parser.add_argument(
        "--news-data-dir",
        default="data/raw/news/mubasher_sentiment_only",
        help="Raw Mubasher JSONL cache/output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sentiment_only_pipeline",
        help="Pipeline artifact output directory.",
    )
    parser.add_argument(
        "--processed-news-dir",
        default="data/processed/news/sentiment_ready",
        help="Clean sentiment-ready news dataset output directory.",
    )
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum Mubasher pages per ticker/language.")
    parser.add_argument("--price-period", default="10y", help="yfinance period for price downloads.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days.")
    parser.add_argument(
        "--min-match-confidence",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum entity-match confidence for articles kept in the benchmark.",
    )
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[3, 7, 14],
        help="Rolling calendar windows used for sentiment features.",
    )
    parser.add_argument("--force-scrape", action="store_true", help="Ignore cached JSONL records and rescrape news.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SentimentOnlyPipeline(
        SentimentOnlyPipelineConfig(
            tickers=tuple(args.tickers),
            news_data_dir=Path(args.news_data_dir),
            processed_news_dir=Path(args.processed_news_dir),
            output_dir=Path(args.output_dir),
            scrape_max_pages=args.max_pages,
            force_scrape=bool(args.force_scrape),
            price_period=args.price_period,
            horizon_days=args.horizon,
            min_match_confidence=args.min_match_confidence,
            rolling_windows=tuple(args.rolling_windows),
        )
    )
    result = pipeline.run()
    print("Saved outputs to", result["output_dir"])
    print("Summary:", result["summary_path"])
    processed_news = result["summary"].get("artifacts", {}).get("processed_news", {})
    if processed_news:
        print("Sentiment-ready news:", processed_news.get("sentiment_labeled_articles_csv"))
    print("Direction metrics:", result["direction_metrics_path"])
    print("Return metrics:", result["return_metrics_path"])


if __name__ == "__main__":
    main()
