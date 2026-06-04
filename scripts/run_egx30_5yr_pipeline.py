#!/usr/bin/env python3
"""Run the resumable full-universe EGX30 sentiment research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List


CAMELBERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse the complete scrape cache.")
    return parser.parse_args()


def run(command: List[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    python = sys.executable
    run([python, "scripts/build_egx30_historical_constituents.py"])
    if not args.skip_scrape:
        run([python, "scripts/scrape_egx30_5yr_news.py"])
    run([python, "scripts/build_egx30_5yr_processed_news.py"])
    run(
        [
            python,
            "scripts/build_egx30_5yr_processed_news.py",
            "--reuse-prepared",
            "--output-path",
            "data/processed_news_5yr_camelbert.csv",
            "--sentiment-source",
            "finbert_camelbert",
            "--arabic-model-name",
            CAMELBERT_MODEL,
        ]
    )
    run(
        [
            python,
            "scripts/run_egx30_5yr_evaluation.py",
            "--camelbert-processed-news-path",
            "data/processed_news_5yr_camelbert.csv",
        ]
    )
    run([python, "scripts/run_egx30_5yr_validity_checks.py"])
    run([python, "scripts/run_easy_win_validity_extensions.py"])
    run([python, "scripts/write_egx30_5yr_research_summary.py"])
    run([python, "scripts/sync_egx30_5yr_docs.py"])
    run([python, "scripts/sync_egx30_5yr_latex.py"])
    run([python, "scripts/compile_thesis_clean.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
