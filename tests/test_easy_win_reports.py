"""Tests for additive easy-win audit and robustness reports."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from egx_direction.easy_win_reports import (
    abstention_rows_for_panel,
    baseline_rows_for_panel,
    build_sector_daily,
    company_coverage,
    daily_coverage,
    yearly_coverage,
    write_weighting_report,
)


class EasyWinReportsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.articles = pd.DataFrame(
            [
                {"Ticker": "AAA", "Date": "2024-01-02", "detected_language": "en", "sentiment_score": 0.3, "confidence": 0.8},
                {"Ticker": "AAA", "Date": "2024-01-03", "detected_language": "ar", "sentiment_score": -0.1, "confidence": 0.7},
                {"Ticker": "BBB", "Date": "2024-01-03", "detected_language": "en", "sentiment_score": 0.2, "confidence": 0.9},
            ]
        )
        self.constituents = pd.DataFrame(
            [
                {"Ticker": "AAA", "Company_Name": "Alpha Bank", "Date_From": "2024-01-01", "Date_To": "2024-01-10"},
                {"Ticker": "BBB", "Company_Name": "Beta Foods", "Date_From": "2024-01-01", "Date_To": "2024-01-10"},
            ]
        )

    def test_coverage_outputs_expected_columns_and_sums(self) -> None:
        trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        company = company_coverage(self.articles, self.constituents)
        yearly = yearly_coverage(self.articles, trading_dates)
        daily = daily_coverage(self.articles, trading_dates)

        self.assertIn("arabic_article_count", company.columns)
        self.assertIn("covered_trading_day_pct", yearly.columns)
        self.assertIn("distinct_companies_covered", daily.columns)
        self.assertEqual(int(company["article_count"].sum()), len(self.articles))
        self.assertEqual(int(yearly["article_count"].sum()), len(self.articles))

    def test_baseline_predictions_have_expected_behavior(self) -> None:
        panel = self._panel(90)
        predictions = self._predictions(panel, horizon=1)
        rows = baseline_rows_for_panel(
            panel,
            predictions,
            horizon=1,
            sentiment_source="source",
            view="bilingual",
            selected_c=0.1,
            random_seed=7,
            refit_every_rows=10,
        )
        test_rows = rows[rows["split"] == "test"]
        self.assertEqual(float(test_rows[test_rows["baseline_name"] == "always_up"]["predicted_up_share"].iloc[0]), 1.0)
        self.assertEqual(float(test_rows[test_rows["baseline_name"] == "always_down"]["predicted_up_share"].iloc[0]), 0.0)
        self.assertIn("article_count_only_logit", set(test_rows["baseline_name"]))
        self.assertIn("sentiment_shuffled_logit", set(test_rows["baseline_name"]))

    def test_abstention_uses_validation_threshold_and_reports_coverage(self) -> None:
        panel = self._panel(90)
        predictions = self._predictions(panel, horizon=1)
        rows = abstention_rows_for_panel(panel, predictions, horizon=1, sentiment_source="source", view="bilingual")
        self.assertTrue((rows["covered_test_samples"] + rows["abstained_test_samples"] > 0).all())
        self.assertTrue(((rows["coverage_pct"] >= 0.0) & (rows["coverage_pct"] <= 1.0)).all())
        self.assertIn("training_selected_threshold", rows.columns)

    def test_sector_builder_tolerates_unknown_sector(self) -> None:
        sector_map = pd.DataFrame(
            [{"ticker": "AAA", "company_name": "Alpha Bank", "sector": "Banks", "source_note": "test"}]
        )
        sector_daily = build_sector_daily(self.articles, sector_map)
        self.assertIn("UNKNOWN", set(sector_daily["sector"]))
        self.assertIn("sector_confidence_weighted_sentiment", sector_daily.columns)

    def test_weighting_report_marks_index_weights_unavailable(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            summary = write_weighting_report(self.articles, Path(tmp))
        self.assertFalse(summary["index_weights_available"])

    @staticmethod
    def _panel(rows: int) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=rows, freq="B")
        target = ([0, 1, 1, 0, 1] * ((rows // 5) + 1))[:rows]
        return pd.DataFrame(
            {
                "Date": dates,
                "Ticker": "EGX30",
                "Close": 100.0,
                "target_return": [0.01 if value else -0.01 for value in target],
                "target_direction": target,
                "target_horizon_end_date": dates,
                "combined_article_count_lag1d": [idx % 4 for idx in range(rows)],
                "combined_sentiment_weighted_mean_lag1d": [((idx % 7) - 3) / 10 for idx in range(rows)],
                "constituent_coverage_ratio_lag1d": [0.2 + (idx % 3) * 0.2 for idx in range(rows)],
            }
        )

    @staticmethod
    def _predictions(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = []
        for split, start, end in (("validation", 59, 72), ("test", 77, len(panel))):
            for idx in range(start, end):
                actual = int(panel.iloc[idx]["target_direction"])
                rows.append(
                    {
                        "horizon": horizon,
                        "sentiment_source": "source",
                        "view": "bilingual",
                        "Date": panel.iloc[idx]["Date"],
                        "Ticker": "EGX30",
                        "split": split,
                        "row_index": idx,
                        "actual_direction": actual,
                        "actual_return": float(panel.iloc[idx]["target_return"]),
                        "predicted_direction": actual,
                        "predicted_probability": 0.8 if actual else 0.2,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
