"""Generate a resumable GPT-4o-mini zero-shot sentiment cache for EGX articles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egx_direction.llm_sentiment import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    DEFAULT_PROMPT_VERSION,
    OpenAIZeroShotSentimentClassifier,
    generate_llm_sentiment_cache,
)
from egx_direction.market_features import scope_articles_to_market  # noqa: E402
from egx_direction.sentiment_only_pipeline import SentimentOnlyPipelineConfig  # noqa: E402
from run_sentiment_only_evaluation_from_dataset import load_scored_articles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached GPT zero-shot sentiment for the EGX30 comparison.")
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
    )
    parser.add_argument("--output-path", default="data/processed/news/sentiment_ready/gpt4o_mini_sentiment_cache.csv")
    parser.add_argument("--news-tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--max-articles", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to generate the GPT zero-shot sentiment cache.")
    config = SentimentOnlyPipelineConfig(tickers=tuple(args.news_tickers))
    articles = scope_articles_to_market(load_scored_articles(Path(args.sentiment_labeled_path), config))
    classifier = OpenAIZeroShotSentimentClassifier(model=args.model)
    cache = generate_llm_sentiment_cache(
        articles,
        output_path=Path(args.output_path),
        classifier=classifier,
        prompt_version=args.prompt_version,
        max_articles=args.max_articles,
    )
    print(f"Cached {len(cache)} article-level zero-shot sentiment rows at {args.output_path}")


if __name__ == "__main__":
    main()
