#!/usr/bin/env python3
"""Prepare and score the consolidated five-year bilingual EGX30 article corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.pit_constituents import EGX30_RESEARCH_END, EGX30_RESEARCH_START  # noqa: E402
from egx_direction.sentiment_only_pipeline import (  # noqa: E402
    SentimentOnlyPipelineConfig,
    prepare_sentiment_articles,
    score_prepared_articles,
)
from news_pipeline import FinBERTConfig  # noqa: E402
from news_pipeline.storage import load_jsonl_records  # noqa: E402


DEFAULT_RAW_PATH = Path("data/scraped_cache/egx30_5yr/all_news.jsonl")
DEFAULT_PREPARED_PATH = Path("data/processed_news_5yr.prepared.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed_news_5yr.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--prepared-path", type=Path, default=DEFAULT_PREPARED_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--sentiment-source", default="finbert_arabert")
    parser.add_argument("--arabic-model-name", default=FinBERTConfig().arabic_model_name)
    parser.add_argument("--skip-state-update", action="store_true", help="Do not mutate data/pipeline_state.json during smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_state_update:
        update_pipeline_state(current_phase="phase_3_article_scoring", status="in_progress")
    if args.reuse_prepared and args.prepared_path.exists():
        prepared = pd.read_csv(args.prepared_path)
    else:
        raw = pd.DataFrame(load_jsonl_records(args.raw_path))
        if raw.empty:
            raise ValueError(f"No scraped articles found: {args.raw_path}")
        prepared = prepare_sentiment_articles(raw, timezone="Africa/Cairo", min_match_confidence="low")
        prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce").dt.normalize()
        prepared = prepared[
            prepared["Date"].between(
                pd.Timestamp(EGX30_RESEARCH_START),
                pd.Timestamp(EGX30_RESEARCH_END),
                inclusive="both",
            )
        ].copy()
        args.prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(args.prepared_path, index=False)

    config = SentimentOnlyPipelineConfig(
        min_match_confidence="low",
        finbert_config=FinBERTConfig(arabic_model_name=args.arabic_model_name),
    )
    scored = score_prepared_articles(prepared, config)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_path, index=False)
    summary = {
        "raw_path": str(args.raw_path),
        "prepared_path": str(args.prepared_path),
        "output_path": str(args.output_path),
        "sentiment_source": args.sentiment_source,
        "arabic_model_name": args.arabic_model_name,
        "prepared_articles": int(len(prepared)),
        "scored_articles": int(len(scored)),
        "language_counts": scored["detected_language"].value_counts(dropna=False).to_dict(),
    }
    if not args.skip_state_update:
        update_pipeline_state(
            current_phase="phase_3_article_scoring",
            status="completed",
            completed_phases=["phase_3_article_scoring"],
            artifacts={
                f"processed_news_{args.sentiment_source}": str(args.output_path),
                f"processed_news_{args.sentiment_source}_summary": summary,
            },
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
