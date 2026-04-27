from __future__ import annotations

import unittest

import pandas as pd

from egx_direction.event_features import (
    aggregate_daily_event_features,
    build_event_feature_panel,
    label_event_articles,
)


class EventFeatureTests(unittest.TestCase):
    def test_label_event_articles_matches_english_and_arabic_keywords(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-01"),
                    "detected_language": "en",
                    "clean_article_text": "CIB reports higher net profit and revenue.",
                    "is_scored": 1,
                    "sentiment_label": "positive",
                    "sentiment_score_standardized": 0.7,
                    "sentiment_article_weight": 0.9,
                },
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-01"),
                    "detected_language": "ar",
                    "clean_article_text": "الشركة تعلن القوائم المالية وتوزيعات نقدية.",
                    "is_scored": 1,
                    "sentiment_label": "neutral",
                    "sentiment_score_standardized": 0.0,
                    "sentiment_article_weight": 0.8,
                },
            ]
        )

        labeled = label_event_articles(articles)

        self.assertTrue(labeled["event_flag"].all())
        self.assertIn("earnings", set(labeled["event_category"]))
        self.assertIn("dividends", set(labeled["event_category"]))
        self.assertGreater(labeled["event_weight"].min(), 1.0)

    def test_event_daily_features_and_panel_are_lagged(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "Ticker": "CIB",
                    "Date": pd.Timestamp("2026-01-01"),
                    "detected_language": "en",
                    "clean_article_text": "CIB earnings results were positive.",
                    "is_scored": 1,
                    "sentiment_label": "positive",
                    "sentiment_score_standardized": 0.5,
                    "sentiment_article_weight": 1.0,
                }
            ]
        )
        labeled = label_event_articles(articles)
        daily = aggregate_daily_event_features(labeled, view="bilingual", strategy="event_only")
        price_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "Ticker": ["CIB", "CIB", "CIB"],
                "Close": [100.0, 101.0, 102.0],
            }
        )

        panel, features = build_event_feature_panel(
            price_df=price_df,
            daily_features=daily,
            horizon_days=1,
            rolling_windows=(3,),
        )

        self.assertIn("event_article_count", features)
        self.assertEqual(panel.loc[0, "event_article_count"], 0.0)
        self.assertEqual(panel.loc[1, "event_article_count"], 1.0)
        self.assertAlmostEqual(panel.loc[1, "event_sentiment_weighted_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
