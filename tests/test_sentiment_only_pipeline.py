from __future__ import annotations

import unittest

import pandas as pd

from egx_direction.sentiment_only_pipeline import (
    aggregate_daily_sentiment_features,
    align_sentiment_features_to_prices,
    build_time_series_split,
    prepare_sentiment_articles,
)


class SentimentOnlyPipelineTests(unittest.TestCase):
    def test_prepare_sentiment_articles_filters_low_confidence_and_scores_relevance(self) -> None:
        raw_news = pd.DataFrame(
            [
                {
                    "ticker": "COMI.CA",
                    "published_at": "2026-01-03T10:00:00+00:00",
                    "article_id": "1",
                    "article_url": "https://example.com/1",
                    "article_language": "en",
                    "match_confidence": "high",
                    "match_methods": ["stock_page", "title_alias"],
                    "matched_primary_aliases": ["CIB"],
                    "matched_secondary_aliases": [],
                    "article_title": "CIB launches new product",
                    "article_summary": "Summary",
                    "article_body": "Body",
                    "is_paywalled": False,
                },
                {
                    "ticker": "CIB",
                    "published_at": "2026-01-04T10:00:00+00:00",
                    "article_id": "2",
                    "article_url": "https://example.com/2",
                    "article_language": "ar",
                    "match_confidence": "low",
                    "match_methods": ["stock_page"],
                    "matched_primary_aliases": [],
                    "matched_secondary_aliases": [],
                    "article_title": "Noise",
                    "article_summary": "Summary",
                    "article_body": "Body",
                    "is_paywalled": False,
                },
            ]
        )

        prepared = prepare_sentiment_articles(
            raw_news=raw_news,
            timezone="Africa/Cairo",
            min_match_confidence="medium",
        )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.iloc[0]["Ticker"], "CIB")
        self.assertEqual(prepared.iloc[0]["entity_relevance_label"], "direct")
        self.assertGreater(prepared.iloc[0]["entity_relevance_weight"], 0.9)

    def test_aggregate_daily_sentiment_features_combines_counts_and_scores(self) -> None:
        scored_articles = pd.DataFrame(
            [
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-03"),
                    "detected_language": "en",
                    "is_scored": 1,
                    "sentiment_label": "positive",
                    "sentiment_article_weight": 0.8,
                    "entity_relevance_label": "direct",
                    "entity_relevance_weight": 1.0,
                    "confidence": 0.8,
                    "sentiment_score_standardized": 0.6,
                    "positive_prob": 0.7,
                    "negative_prob": 0.1,
                    "neutral_prob": 0.2,
                },
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-03"),
                    "detected_language": "ar",
                    "is_scored": 1,
                    "sentiment_label": "negative",
                    "sentiment_article_weight": 0.4,
                    "entity_relevance_label": "contextual",
                    "entity_relevance_weight": 0.5,
                    "confidence": 0.8,
                    "sentiment_score_standardized": -0.3,
                    "positive_prob": 0.2,
                    "negative_prob": 0.6,
                    "neutral_prob": 0.2,
                },
            ]
        )

        daily = aggregate_daily_sentiment_features(scored_articles=scored_articles, view="bilingual")

        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.iloc[0]["article_count_total"], 2.0)
        self.assertEqual(daily.iloc[0]["article_count_positive"], 1.0)
        self.assertEqual(daily.iloc[0]["article_count_negative"], 1.0)
        self.assertAlmostEqual(daily.iloc[0]["sentiment_score_mean"], 0.15)
        self.assertAlmostEqual(daily.iloc[0]["sentiment_score_weighted_mean"], 0.3)
        self.assertEqual(daily.iloc[0]["article_count_en"], 1.0)
        self.assertEqual(daily.iloc[0]["article_count_ar"], 1.0)

    def test_align_sentiment_features_to_prices_uses_previous_day_lag(self) -> None:
        price_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
                "Ticker": ["CIB", "CIB", "CIB"],
                "Close": [100.0, 101.0, 102.0],
            }
        )
        daily_features = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "Ticker": ["CIB", "CIB", "CIB"],
                "article_count_total": [2.0, 1.0, 0.0],
                "sentiment_score_mean": [0.5, -0.2, 0.0],
            }
        )

        aligned = align_sentiment_features_to_prices(
            price_df=price_df,
            daily_features=daily_features,
            rolling_windows=(3,),
            lag_days=1,
        )

        self.assertEqual(list(aligned["article_count_total"]), [2.0, 1.0, 0.0])
        self.assertAlmostEqual(aligned.iloc[0]["sentiment_score_mean"], 0.5)
        self.assertAlmostEqual(aligned.iloc[1]["sentiment_score_mean"], -0.2)
        self.assertIn("article_count_total_roll3d", aligned.columns)
        self.assertIn("days_since_article", aligned.columns)

    def test_build_time_series_split_respects_min_rows_and_embargo(self) -> None:
        split = build_time_series_split(
            n_rows=900,
            split_config=type("Cfg", (), {"validation_ratio": 0.2, "test_ratio": 0.2, "min_train_rows": 250, "embargo_rows": 5})(),
            horizon_days=5,
        )

        self.assertGreaterEqual(split.train_end, 250)
        self.assertEqual(split.validation_start, split.train_end + 5)
        self.assertEqual(split.test_start, split.validation_end + 5)
        self.assertGreater(len(split.validation_indices), 0)
        self.assertGreater(len(split.test_indices), 0)


if __name__ == "__main__":
    unittest.main()
