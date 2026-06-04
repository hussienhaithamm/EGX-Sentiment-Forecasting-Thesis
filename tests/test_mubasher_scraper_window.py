"""Tests for timeline filtering in the resumable Mubasher scraper."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from news_pipeline import AlMobasherScraper, AlMobasherScraperConfig, TargetCompany
from news_pipeline.storage import load_jsonl_records, write_jsonl_records


class MubasherScraperWindowTests(unittest.TestCase):
    def test_resume_cache_is_filtered_to_research_window(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scraper = AlMobasherScraper(
                config=AlMobasherScraperConfig(
                    output_dir=root / "cache",
                    log_dir=root / "logs",
                    start_date="2021-04-21",
                    end_date="2026-04-20",
                ),
                tickers=[],
                languages=[],
                companies={},
            )
            write_jsonl_records(
                scraper._ticker_output_path("TEST"),
                [
                    {"article_id": "before", "article_language": "en", "published_at": "2021-04-20T12:00:00+00:00"},
                    {"article_id": "inside", "article_language": "en", "published_at": "2021-04-21T12:00:00+00:00"},
                    {"article_id": "after", "article_language": "en", "published_at": "2026-04-21T12:00:00+00:00"},
                ],
            )
            records = scraper._load_existing_records("TEST")
            self.assertEqual(set(records), {("inside", "en")})

    def test_language_checkpoint_is_bound_to_timeline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AlMobasherScraperConfig(
                output_dir=root / "cache",
                log_dir=root / "logs",
                start_date="2021-04-21",
                end_date="2026-04-20",
            )
            scraper = AlMobasherScraper(config=config, tickers=[], languages=[], companies={})
            scraper._mark_language_complete("TEST", "en")
            self.assertEqual(scraper._load_completed_languages("TEST"), {"en"})

            changed = AlMobasherScraper(
                config=AlMobasherScraperConfig(
                    output_dir=root / "cache",
                    log_dir=root / "changed-logs",
                    start_date="2021-04-21",
                    end_date="2026-04-19",
                ),
                tickers=[],
                languages=[],
                companies={},
            )
            self.assertEqual(changed._load_completed_languages("TEST"), set())

    def test_targeted_resume_consolidates_all_company_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            companies = {
                ticker: TargetCompany(ticker=ticker, site_code=ticker, company_name_en=ticker, company_name_ar="")
                for ticker in ("AAA", "BBB")
            }
            scraper = AlMobasherScraper(
                config=AlMobasherScraperConfig(output_dir=root / "cache", log_dir=root / "logs"),
                tickers=["AAA"],
                languages=[],
                companies=companies,
            )
            for ticker in companies:
                write_jsonl_records(
                    scraper._ticker_output_path(ticker),
                    [{"article_id": ticker, "article_language": "en", "published_at": "2024-01-01T00:00:00+00:00"}],
                )
            scraper._persist_combined_news()
            records = load_jsonl_records(root / "cache" / "all_news.jsonl")
            self.assertEqual({row["article_id"] for row in records}, {"AAA", "BBB"})


if __name__ == "__main__":
    unittest.main()
