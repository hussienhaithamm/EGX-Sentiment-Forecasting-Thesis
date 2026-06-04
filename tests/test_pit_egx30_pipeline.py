"""Tests for the full-universe point-in-time EGX30 feature layer."""

from __future__ import annotations

import unittest

import pandas as pd

from egx_direction.pit_constituents import active_tickers, validate_constituent_registry
from egx_direction.pit_market_features import aggregate_equal_weight_market_daily, scope_articles_to_pit_market


class PitEgx30PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for ticker in ("AAA", "BBB"):
            rows.append(
                {
                    "Date_From": "2021-04-21",
                    "Date_To": "2021-04-22",
                    "Ticker": ticker,
                    "Mubasher_Slug": ticker,
                    "Company_Name": ticker,
                }
            )
        self.constituents = pd.DataFrame(rows)

    def test_active_tickers_are_point_in_time(self) -> None:
        self.assertEqual(active_tickers(self.constituents, "2021-04-21"), ("AAA", "BBB"))

    def test_scope_drops_inactive_and_cross_company_duplicate_articles(self) -> None:
        articles = pd.DataFrame(
            [
                {"Ticker": "AAA", "Date": "2021-04-21", "article_id": "1", "detected_language": "en", "sentiment_article_weight": 0.2},
                {"Ticker": "BBB", "Date": "2021-04-21", "article_id": "1", "detected_language": "en", "sentiment_article_weight": 0.8},
                {"Ticker": "CCC", "Date": "2021-04-21", "article_id": "2", "detected_language": "en", "sentiment_article_weight": 1.0},
            ]
        )
        scoped = scope_articles_to_pit_market(articles, self.constituents)
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped.iloc[0]["Ticker"], "BBB")
        self.assertEqual(scoped.iloc[0]["Source_Tickers"], "AAA|BBB")

    def test_equal_weight_market_mood_includes_zero_for_uncovered_company(self) -> None:
        company_daily = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Date": "2021-04-21",
                    "combined_article_count": 2.0,
                    "combined_sentiment_weighted_mean": 0.8,
                }
            ]
        )
        market = aggregate_equal_weight_market_daily(
            company_daily,
            self.constituents,
            start_date="2021-04-21",
            end_date="2021-04-21",
        )
        self.assertAlmostEqual(market.iloc[0]["combined_sentiment_weighted_mean"], 0.4)
        self.assertEqual(market.iloc[0]["combined_article_count"], 2.0)
        self.assertAlmostEqual(market.iloc[0]["constituent_coverage_ratio"], 0.5)

    def test_real_registry_validator_requires_documented_counts(self) -> None:
        with self.assertRaises(ValueError):
            validate_constituent_registry(self.constituents)


if __name__ == "__main__":
    unittest.main()
