#!/usr/bin/env python3
"""Resumably scrape bilingual Mubasher news for the historical EGX30 company union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.pit_constituents import (  # noqa: E402
    DEFAULT_COMPANY_PATH,
    DEFAULT_CONSTITUENT_PATH,
    EGX30_RESEARCH_END,
    EGX30_RESEARCH_START,
    load_constituent_registry,
    load_mubasher_companies,
    validate_constituent_registry,
)
from news_pipeline import AlMobasherScraper, AlMobasherScraperConfig  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data/scraped_cache/egx30_5yr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--company-registry", type=Path, default=DEFAULT_COMPANY_PATH)
    parser.add_argument("--constituent-registry", type=Path, default=DEFAULT_CONSTITUENT_PATH)
    parser.add_argument("--tickers", nargs="+", help="Optional subset for a smoke test or targeted resume.")
    parser.add_argument("--languages", nargs="+", choices=["en", "ar"], default=["en", "ar"])
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay-min", type=float, default=0.5)
    parser.add_argument("--delay-max", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delay_min < 0.5 or args.delay_max > 1.5 or args.delay_min > args.delay_max:
        raise ValueError("Politeness delay must stay within the required randomized 0.5-1.5 second range.")

    constituents = load_constituent_registry(args.constituent_registry)
    validation = validate_constituent_registry(constituents)
    companies = load_mubasher_companies(args.company_registry)
    tickers = [value.upper() for value in (args.tickers or sorted(companies))]
    unknown = sorted(set(tickers) - set(companies))
    if unknown:
        raise ValueError(f"Unknown company identities: {unknown}")

    update_pipeline_state(
        current_phase="phase_2_dynamic_news_scrape",
        status="in_progress",
        artifacts={"scraped_cache": str(args.output_dir)},
        notes=["Full-universe scraping is resumable by company JSONL files and all-news manifest."],
    )
    config = AlMobasherScraperConfig(
        output_dir=args.output_dir,
        log_dir=Path("logs/news/egx30_5yr"),
        request_delay_seconds=args.delay_min,
        request_delay_min_seconds=args.delay_min,
        request_delay_max_seconds=args.delay_max,
        max_retries=4,
        backoff_factor=1.5,
        max_pages_per_language=args.max_pages,
        stop_after_seen_pages=0,
        stop_after_before_start_pages=1,
        start_date=EGX30_RESEARCH_START,
        end_date=EGX30_RESEARCH_END,
        force_rescrape=args.force,
        include_raw_html=False,
    )
    scraper = AlMobasherScraper(
        config=config,
        tickers=tickers,
        languages=args.languages,
        companies=companies,
    )
    scraper.run(workers=args.workers)
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    complete_full_union = set(tickers) == set(companies) and set(args.languages) == {"en", "ar"}
    update_pipeline_state(
        current_phase="phase_2_dynamic_news_scrape",
        status="completed" if complete_full_union else "in_progress",
        completed_phases=["phase_2_dynamic_news_scrape"] if complete_full_union else [],
        artifacts={
            "scraped_cache": str(args.output_dir),
            "scrape_manifest": str(args.output_dir / "manifest.json"),
            "scraped_company_count": len(manifest["summary"]),
        },
    )
    print(json.dumps({"validation": validation["status"], "companies": len(tickers), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
