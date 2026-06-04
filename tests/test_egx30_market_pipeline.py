from __future__ import annotations

import unittest

import pandas as pd

from egx_direction.market_features import (
    aggregate_daily_market_scope_features,
    scope_articles_to_market,
)
from egx_direction.market_index import (
    EGX30_TICKER,
    fetch_mubasher_egx30_prices,
    normalize_egx30_payload,
    validate_egx30_prices,
)
from egx_direction.sentiment_only_pipeline import SentimentOnlyPipelineConfig
from scripts.run_sentiment_only_evaluation_from_dataset import build_enhanced_sentiment_panel


class Egx30MarketPipelineTests(unittest.TestCase):
    def test_fetch_mubasher_index_csv_normalizes_and_deduplicates_dates(self) -> None:
        class FakeResponse:
            text = "\n".join(
                [
                    "2026-01-01/00:00:00,99,102,98,100,1000",
                    "2026-01-01/00:00:00,99,103,98,100,1100",
                    "2026-01-02/00:00:00,100,104,99,103,1200",
                ]
            )

            @staticmethod
            def raise_for_status() -> None:
                return None

        class FakeSession:
            @staticmethod
            def get(*args, **kwargs) -> FakeResponse:
                return FakeResponse()

        prices = fetch_mubasher_egx30_prices(
            start_date="2026-01-01",
            end_date="2026-01-02",
            session=FakeSession(),
        )

        self.assertEqual(prices["Date"].dt.date.astype(str).tolist(), ["2026-01-01", "2026-01-02"])
        self.assertEqual(prices["Close"].tolist(), [100, 103])

    def test_normalize_official_index_payload_filters_timeline(self) -> None:
        payload = [
            {"CDAY": "2026-01-01", "INDEX_VALUE": "30000.5"},
            {"CDAY": "2026-01-02", "INDEX_VALUE": "30,100.25"},
            {"CDAY": "2026-01-03", "INDEX_VALUE": "30200"},
        ]

        normalized = normalize_egx30_payload(
            payload,
            start_date="2026-01-02",
            end_date="2026-01-03",
        )

        self.assertEqual(normalized["Ticker"].tolist(), [EGX30_TICKER, EGX30_TICKER])
        self.assertEqual(normalized["Close"].tolist(), [30100.25, 30200.0])

    def test_validate_index_prices_rejects_duplicate_dates(self) -> None:
        prices = pd.DataFrame(
            {
                "Date": ["2026-01-01", "2026-01-01"],
                "Close": [30000.0, 30100.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate trading dates"):
            validate_egx30_prices(prices, start_date="2026-01-01", end_date="2026-01-02")

    def test_market_scope_deduplicates_cross_company_articles(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-01"),
                    "published_at": pd.Timestamp("2026-01-01 10:00:00"),
                    "article_id": "same-story",
                    "detected_language": "en",
                    "sentiment_article_weight": 0.5,
                },
                {
                    "Ticker": "TMG",
                    "Date": pd.Timestamp("2026-01-01"),
                    "published_at": pd.Timestamp("2026-01-01 10:00:00"),
                    "article_id": "same-story",
                    "detected_language": "en",
                    "sentiment_article_weight": 0.8,
                },
            ]
        )

        scoped = scope_articles_to_market(articles)

        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped.iloc[0]["Ticker"], EGX30_TICKER)
        self.assertEqual(scoped.iloc[0]["SourceTicker"], "MULTI")
        self.assertEqual(scoped.iloc[0]["SourceTickers"], "CIB|TMG")
        self.assertEqual(scoped.iloc[0]["source_company_count"], 2.0)
        self.assertEqual(scoped.iloc[0]["sentiment_article_weight"], 0.8)

    def test_daily_market_scope_features_count_firm_and_macro_news(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "Ticker": EGX30_TICKER,
                    "Date": pd.Timestamp("2026-01-01"),
                    "detected_language": "en",
                    "SourceTickers": "CIB|TMG",
                    "NewsScope": "firm",
                },
                {
                    "Ticker": EGX30_TICKER,
                    "Date": pd.Timestamp("2026-01-01"),
                    "detected_language": "ar",
                    "SourceTickers": "EGX30",
                    "NewsScope": "macro",
                },
            ]
        )

        daily = aggregate_daily_market_scope_features(articles, view="bilingual")

        self.assertEqual(daily.iloc[0]["source_company_count"], 2.0)
        self.assertEqual(daily.iloc[0]["source_cib_article_count"], 1.0)
        self.assertEqual(daily.iloc[0]["source_tmg_article_count"], 1.0)
        self.assertEqual(daily.iloc[0]["firm_article_count"], 1.0)
        self.assertEqual(daily.iloc[0]["macro_article_count"], 1.0)

    def test_egx30_closes_build_labels_but_do_not_enter_features(self) -> None:
        prices = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
                "Ticker": [EGX30_TICKER] * 4,
                "Close": [100.0, 101.0, 99.0, 102.0],
            }
        )
        daily_features = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "Ticker": [EGX30_TICKER, EGX30_TICKER],
                "source_company_count": [2.0, 1.0],
            }
        )
        config = SentimentOnlyPipelineConfig(
            tickers=(EGX30_TICKER,),
            horizon_days=1,
            rolling_windows=(3,),
        )

        panel, feature_columns = build_enhanced_sentiment_panel(prices, daily_features, config)

        self.assertEqual(panel["target_direction"].tolist(), [1, 0, 1])
        self.assertEqual(panel["source_company_count_lag1d"].tolist(), [0.0, 2.0, 1.0])
        self.assertTrue(
            {"Close", "target_return", "target_direction", "target_horizon_end_date"}.isdisjoint(feature_columns)
        )


if __name__ == "__main__":
    unittest.main()
