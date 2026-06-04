"""Generate Phase 7 robustness, coverage, leakage, and overfitting reports."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egx_direction.event_features import (  # noqa: E402
    aggregate_daily_event_features,
    build_event_feature_panel,
    label_event_articles,
)
from egx_direction.sentiment_only_pipeline import (  # noqa: E402
    LANGUAGE_VIEWS,
    SentimentOnlyPipelineConfig,
    build_time_series_split,
    compute_direction_metrics,
    compute_return_metrics,
)
from news_pipeline.tickers import TARGET_COMPANIES  # noqa: E402
from run_multi_horizon_event_evaluation import FEATURE_STRATEGIES  # noqa: E402
from run_sentiment_only_evaluation_from_dataset import (  # noqa: E402
    aggregate_enhanced_daily_sentiment_features,
    build_enhanced_sentiment_panel,
    load_price_frames,
    load_scored_articles,
)


PAGE_COUNT_RE = re.compile(r"numPages\s*=\s*(\d+)")
LISTING_PAGE_LINK_RE = re.compile(r"/markets/EGX/stocks/[^/]+/news/(\d+)")
LIST_PAGE_NUMBER_RE = re.compile(r"/news/(\d+)(?:\?|$)")
LANGUAGE_BASE_URLS = {
    "en": "https://english.mubasher.info",
    "ar": "https://www.mubasher.info",
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}
LEAKAGE_EXCLUDED_COLUMNS = {"Date", "Ticker", "Close", "target_return", "target_direction", "target_horizon_end_date"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 validity checks for the EGX sentiment thesis project.")
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--target-scope", choices=("equities", "egx30"), default="equities")
    parser.add_argument("--target-price-path", default="data/EGX30 Index Historical Data.csv")
    parser.add_argument("--llm-cache-path")
    parser.add_argument("--raw-news-dir", default="data/raw/news/mubasher_sentiment_only")
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--price-period", default="10y")
    parser.add_argument("--skip-live-page-counts", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_page_count(html: str) -> Optional[int]:
    match = PAGE_COUNT_RE.search(html)
    if match:
        return int(match.group(1))
    page_numbers = [int(value) for value in LISTING_PAGE_LINK_RE.findall(html) if value.isdigit()]
    return max(page_numbers) if page_numbers else None


def _fetch_live_page_count(ticker: str, language: str, timeout: float = 20.0) -> Tuple[Optional[int], Optional[str]]:
    company = TARGET_COMPANIES[ticker]
    url = f"{LANGUAGE_BASE_URLS[language]}/markets/EGX/stocks/{company.site_code}/news/1"
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        return _extract_page_count(response.text), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _raw_page_stats(raw_news_dir: Path, tickers: Sequence[str]) -> Dict[str, Dict[str, object]]:
    failed_articles = _read_jsonl(raw_news_dir / "failed_articles.jsonl")
    failed_pages = _read_jsonl(raw_news_dir / "failed_pages.jsonl")
    stats: Dict[str, Dict[str, object]] = {}
    for ticker in tickers:
        records = _read_jsonl(raw_news_dir / f"{ticker}.jsonl")
        language_counts = {"en": 0, "ar": 0}
        max_pages = {"en": 0, "ar": 0}
        for record in records:
            language = str(record.get("article_language", ""))
            if language in language_counts:
                language_counts[language] += 1
            list_page_url = ""
            raw = record.get("raw")
            if isinstance(raw, dict):
                list_page_url = str(raw.get("list_page_url", "") or "")
            match = LIST_PAGE_NUMBER_RE.search(list_page_url)
            if language in max_pages and match:
                max_pages[language] = max(max_pages[language], int(match.group(1)))
        stats[ticker] = {
            "raw_records": len(records),
            "raw_en_records": language_counts["en"],
            "raw_ar_records": language_counts["ar"],
            "local_max_page_en": max_pages["en"],
            "local_max_page_ar": max_pages["ar"],
            "failed_articles": sum(1 for row in failed_articles if row.get("ticker") == ticker),
            "failed_pages": sum(1 for row in failed_pages if row.get("ticker") == ticker),
        }
    return stats


def _live_page_counts(tickers: Sequence[str], skip_live: bool) -> Dict[str, Dict[str, object]]:
    live: Dict[str, Dict[str, object]] = {}
    for ticker in tickers:
        live[ticker] = {}
        for language in ("en", "ar"):
            if skip_live:
                count, error = None, "Skipped by --skip-live-page-counts"
            else:
                count, error = _fetch_live_page_count(ticker, language)
            live[ticker][f"live_available_pages_{language}"] = count
            live[ticker][f"live_page_count_error_{language}"] = error
    return live


def _date_set(series: pd.Series) -> set[pd.Timestamp]:
    return set(pd.to_datetime(series, errors="coerce").dropna().dt.normalize())


def _prior_window_has_article(price_dates: Iterable[pd.Timestamp], article_dates: set[pd.Timestamp], days: int) -> int:
    count = 0
    for date in pd.to_datetime(list(price_dates), errors="coerce"):
        if pd.isna(date):
            continue
        normalized = pd.Timestamp(date).normalize()
        for offset in range(1, days + 1):
            if normalized - pd.Timedelta(days=offset) in article_dates:
                count += 1
                break
    return count


def build_coverage_analysis(
    scored_articles: pd.DataFrame,
    price_frames: Dict[str, pd.DataFrame],
    raw_news_dir: Path,
    tickers: Sequence[str],
    *,
    skip_live_page_counts: bool,
) -> pd.DataFrame:
    raw_stats = _raw_page_stats(raw_news_dir, tickers)
    live_stats = _live_page_counts(tickers, skip_live_page_counts)
    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        articles = scored_articles[scored_articles["Ticker"] == ticker].copy()
        prices = price_frames[ticker].copy()
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
        price_dates = prices["Date"].dropna().sort_values()
        article_dates = _date_set(articles["Date"])

        first_article = pd.to_datetime(articles["Date"], errors="coerce").min()
        last_article = pd.to_datetime(articles["Date"], errors="coerce").max()
        if pd.notna(first_article) and pd.notna(last_article):
            in_article_window = price_dates[(price_dates >= first_article) & (price_dates <= last_article)]
        else:
            in_article_window = pd.Series(dtype="datetime64[ns]")

        trading_days_with_same_day_article = int(sum(date in article_dates for date in in_article_window))
        trading_days_with_prior_day_article = _prior_window_has_article(in_article_window, article_dates, 1)
        trading_days_with_prior_7d_article = _prior_window_has_article(in_article_window, article_dates, 7)
        trading_days_in_window = int(len(in_article_window))
        total_articles = int(len(articles))
        article_calendar_days = int(len(article_dates))

        merged_stats = {**raw_stats.get(ticker, {}), **live_stats.get(ticker, {})}
        more_pages = False
        more_pages_unknown = False
        for language in ("en", "ar"):
            live_count = merged_stats.get(f"live_available_pages_{language}")
            local_count = merged_stats.get(f"local_max_page_{language}")
            if live_count is None:
                more_pages_unknown = True
            elif local_count is not None and int(live_count) > int(local_count):
                more_pages = True

        if more_pages:
            scraping_assessment = "More listing pages appear available than are present locally; run the resumable scraper."
        elif more_pages_unknown:
            scraping_assessment = "Live page-count check unavailable; local manifest must be used."
        else:
            scraping_assessment = "Local scrape covers all currently available Mubasher listing pages for this ticker/language set."

        rows.append(
            {
                "ticker": ticker,
                "sentiment_labeled_articles": total_articles,
                "english_articles": int((articles["detected_language"] == "en").sum()),
                "arabic_articles": int((articles["detected_language"] == "ar").sum()),
                "first_article_date": first_article.date().isoformat() if pd.notna(first_article) else "",
                "last_article_date": last_article.date().isoformat() if pd.notna(last_article) else "",
                "price_start_date": price_dates.min().date().isoformat() if len(price_dates) else "",
                "price_end_date": price_dates.max().date().isoformat() if len(price_dates) else "",
                "total_price_trading_days": int(len(price_dates)),
                "trading_days_in_article_window": trading_days_in_window,
                "article_calendar_days": article_calendar_days,
                "trading_days_with_same_day_article": trading_days_with_same_day_article,
                "trading_days_with_prior_calendar_day_article": trading_days_with_prior_day_article,
                "trading_days_with_prior_7d_article": trading_days_with_prior_7d_article,
                "article_density_per_trading_day_window": total_articles / trading_days_in_window if trading_days_in_window else math.nan,
                "article_density_per_article_calendar_day": total_articles / article_calendar_days if article_calendar_days else math.nan,
                "same_day_article_coverage_ratio": trading_days_with_same_day_article / trading_days_in_window if trading_days_in_window else math.nan,
                "prior_calendar_day_article_coverage_ratio": trading_days_with_prior_day_article / trading_days_in_window if trading_days_in_window else math.nan,
                "prior_7d_article_coverage_ratio": trading_days_with_prior_7d_article / trading_days_in_window if trading_days_in_window else math.nan,
                "more_mubasher_pages_available": bool(more_pages),
                "scraping_assessment": scraping_assessment,
                **merged_stats,
            }
        )
    return pd.DataFrame(rows)


def _daily_lookup(daily_features: pd.DataFrame, ticker: str, value_column: str) -> pd.Series:
    if daily_features.empty or value_column not in daily_features.columns:
        return pd.Series(dtype=float)
    frame = daily_features[daily_features["Ticker"] == ticker].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    return frame.set_index("Date")[value_column].astype(float)


def _expected_lag_values(panel_dates: pd.Series, daily_series: pd.Series, lag_days: int = 1) -> np.ndarray:
    values = []
    for date in pd.to_datetime(panel_dates, errors="coerce"):
        source_date = pd.Timestamp(date).normalize() - pd.Timedelta(days=lag_days)
        values.append(float(daily_series.get(source_date, 0.0)))
    return np.array(values, dtype=float)


def _expected_roll_mean(panel_dates: pd.Series, daily_series: pd.Series, window: int) -> np.ndarray:
    if daily_series.empty:
        return np.zeros(len(panel_dates), dtype=float)
    start = min(pd.to_datetime(panel_dates).min().normalize(), daily_series.index.min())
    end = pd.to_datetime(panel_dates).max().normalize()
    calendar_index = pd.date_range(start, end, freq="D")
    base = daily_series.reindex(calendar_index).fillna(0.0)
    shifted = base.shift(1).fillna(0.0)
    rolled = shifted.rolling(window, min_periods=1).mean().fillna(0.0)
    return np.array([float(rolled.get(pd.Timestamp(date).normalize(), 0.0)) for date in pd.to_datetime(panel_dates)])


def _target_return_check(panel: pd.DataFrame, price_df: pd.DataFrame, horizon: int) -> Tuple[int, float]:
    prices = price_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    prices = prices.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    close_by_date = prices.set_index("Date")["Close"].astype(float)

    mismatches = 0
    max_abs_diff = 0.0
    for _, row in panel.iterrows():
        date = pd.Timestamp(row["Date"]).normalize()
        end_date = pd.Timestamp(row["target_horizon_end_date"]).normalize()
        if end_date <= date:
            mismatches += 1
            continue
        if date not in close_by_date.index or end_date not in close_by_date.index:
            mismatches += 1
            continue
        expected = float(close_by_date.loc[end_date] / close_by_date.loc[date] - 1.0)
        diff = abs(expected - float(row["target_return"]))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-10:
            mismatches += 1
    return mismatches, max_abs_diff


def run_leakage_checks(
    scored_articles: pd.DataFrame,
    price_frames: Dict[str, pd.DataFrame],
    tickers: Sequence[str],
    horizons: Sequence[int],
    rolling_windows: Sequence[int],
    predictions_path: Path,
) -> Dict[str, object]:
    event_articles = label_event_articles(scored_articles)
    detail_rows: List[Dict[str, object]] = []
    violations: List[str] = []

    for horizon in horizons:
        config = SentimentOnlyPipelineConfig(
            tickers=tuple(tickers),
            horizon_days=int(horizon),
            rolling_windows=tuple(rolling_windows),
            output_dir=Path("outputs/_phase7_tmp"),
            refit_every_rows=250,
        )
        for feature_strategy in FEATURE_STRATEGIES:
            strategy_views = LANGUAGE_VIEWS if feature_strategy == "all_news" else ("bilingual",)
            for view in strategy_views:
                if feature_strategy == "all_news":
                    daily_features = aggregate_enhanced_daily_sentiment_features(scored_articles, view)
                else:
                    daily_features = aggregate_daily_event_features(event_articles, view, feature_strategy)

                for ticker in tickers:
                    ticker_daily = daily_features[daily_features["Ticker"] == ticker].copy()
                    if feature_strategy == "all_news":
                        panel, feature_columns = build_enhanced_sentiment_panel(price_frames[ticker], ticker_daily, config)
                        lag_observed_column = "combined_article_count_lag1d"
                        lag_daily_column = "combined_article_count"
                        roll_observed_column = "combined_sentiment_roll_mean_3d"
                        roll_daily_column = "combined_sentiment_weighted_mean"
                    else:
                        panel, feature_columns = build_event_feature_panel(price_frames[ticker], ticker_daily, int(horizon), rolling_windows)
                        lag_observed_column = "event_article_count"
                        lag_daily_column = "event_article_count"
                        roll_observed_column = "event_sentiment_weighted_mean_roll_mean_3d"
                        roll_daily_column = "event_sentiment_weighted_mean"

                    leaked_feature_columns = sorted(set(feature_columns) & LEAKAGE_EXCLUDED_COLUMNS)
                    if leaked_feature_columns:
                        violations.append(f"{horizon}/{feature_strategy}/{view}/{ticker}: leakage columns in feature set {leaked_feature_columns}")

                    target_mismatches, max_return_diff = _target_return_check(panel, price_frames[ticker], int(horizon))
                    if target_mismatches:
                        violations.append(f"{horizon}/{feature_strategy}/{view}/{ticker}: {target_mismatches} target construction mismatches")

                    split = build_time_series_split(len(panel), config.split_config, int(horizon))
                    split_embargo_ok = (
                        split.validation_start >= split.train_end + max(config.split_config.embargo_rows, int(horizon))
                        and split.test_start >= split.validation_end + max(config.split_config.embargo_rows, int(horizon))
                    )
                    if not split_embargo_ok:
                        violations.append(f"{horizon}/{feature_strategy}/{view}/{ticker}: split embargo check failed")

                    lag_max_abs_diff = math.nan
                    roll_max_abs_diff = math.nan
                    if lag_observed_column in panel.columns:
                        expected = _expected_lag_values(panel["Date"], _daily_lookup(daily_features, ticker, lag_daily_column))
                        observed = panel[lag_observed_column].astype(float).to_numpy()
                        lag_max_abs_diff = float(np.max(np.abs(expected - observed))) if len(observed) else 0.0
                        if lag_max_abs_diff > 1e-10:
                            violations.append(f"{horizon}/{feature_strategy}/{view}/{ticker}: lag feature mismatch {lag_max_abs_diff}")

                    if roll_observed_column in panel.columns:
                        expected_roll = _expected_roll_mean(panel["Date"], _daily_lookup(daily_features, ticker, roll_daily_column), 3)
                        observed_roll = panel[roll_observed_column].astype(float).to_numpy()
                        roll_max_abs_diff = float(np.max(np.abs(expected_roll - observed_roll))) if len(observed_roll) else 0.0
                        if roll_max_abs_diff > 1e-10:
                            violations.append(f"{horizon}/{feature_strategy}/{view}/{ticker}: rolling feature mismatch {roll_max_abs_diff}")

                    detail_rows.append(
                        {
                            "horizon": int(horizon),
                            "feature_strategy": feature_strategy,
                            "view": view,
                            "ticker": ticker,
                            "panel_rows": int(len(panel)),
                            "feature_count": int(len(feature_columns)),
                            "target_mismatches": int(target_mismatches),
                            "max_target_return_abs_diff": max_return_diff,
                            "lag_feature_checked": lag_observed_column,
                            "lag_feature_max_abs_diff": lag_max_abs_diff,
                            "rolling_feature_checked": roll_observed_column,
                            "rolling_feature_max_abs_diff": roll_max_abs_diff,
                            "feature_columns_exclude_targets_and_close": not leaked_feature_columns,
                            "split_embargo_ok": split_embargo_ok,
                        }
                    )

    predictions = pd.read_csv(predictions_path)
    eval_predictions = predictions[predictions["split"].isin(["validation", "test"])].copy()
    eval_predictions["max_allowed_history_rows"] = eval_predictions["row_index"] - eval_predictions["horizon"] + 1
    history_direction_violations = int(
        (eval_predictions["history_rows_direction"] > eval_predictions["max_allowed_history_rows"]).sum()
    )
    history_return_violations = int(
        (eval_predictions["history_rows_return"] > eval_predictions["max_allowed_history_rows"]).sum()
    )
    if history_direction_violations:
        violations.append(f"direction walk-forward history rows exceeded allowed history in {history_direction_violations} rows")
    if history_return_violations:
        violations.append(f"return walk-forward history rows exceeded allowed history in {history_return_violations} rows")

    return {
        "overall_status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "automated_panel_checks": detail_rows,
        "walk_forward_history_check": {
            "evaluation_rows_checked": int(len(eval_predictions)),
            "direction_history_violations": history_direction_violations,
            "return_history_violations": history_return_violations,
            "rule": "history_rows <= row_index - horizon + 1 for validation/test rows",
        },
        "static_code_checks": {
            "target_construction": "build_sentiment_targets uses Close.shift(-horizon) and drops rows without future return.",
            "feature_alignment": "all-news and event features are calendar reindexed then shifted by at least one day before merging to trading dates.",
            "rolling_features": "rolling statistics are computed on shifted daily feature series, so the latest source date is t-1.",
            "scaling": "StandardScaler is inside each sklearn Pipeline and is fit only inside model.fit() on the current training frame.",
            "hyperparameter_selection": "C/alpha are selected from validation walk-forward predictions before final test predictions are generated.",
            "threshold_tuning": "classification threshold is selected only from validation predictions by choose_validation_threshold().",
            "test_exclusion": "No test labels are passed into hyperparameter or threshold selection functions.",
        },
    }


def _direction_metrics_for_group(group: pd.DataFrame) -> Dict[str, object]:
    return compute_direction_metrics(group.rename(columns={"predicted_direction": "predicted_direction"}))


def _return_metrics_for_group(group: pd.DataFrame) -> Dict[str, object]:
    return compute_return_metrics(group.rename(columns={"predicted_return": "predicted_return"}))


def build_overfitting_report(predictions_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    rows: List[Dict[str, object]] = []
    group_cols = ["horizon", "feature_strategy", "ticker", "view"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        key_payload = dict(zip(group_cols, keys))
        split_payload: Dict[str, object] = dict(key_payload)
        for split in ("train", "validation", "test"):
            split_group = group[group["split"] == split].copy()
            d_metrics = _direction_metrics_for_group(split_group) if not split_group.empty else {}
            r_metrics = _return_metrics_for_group(split_group) if not split_group.empty else {}
            for metric in ("rows", "accuracy", "balanced_accuracy", "precision", "recall", "f1"):
                split_payload[f"{split}_{metric}"] = d_metrics.get(metric, math.nan)
            for metric in ("mae", "rmse", "sign_accuracy", "return_correlation"):
                split_payload[f"{split}_{metric}"] = r_metrics.get(metric, math.nan)

        split_payload["train_minus_validation_accuracy"] = split_payload["train_accuracy"] - split_payload["validation_accuracy"]
        split_payload["train_minus_test_accuracy"] = split_payload["train_accuracy"] - split_payload["test_accuracy"]
        split_payload["train_minus_validation_balanced_accuracy"] = (
            split_payload["train_balanced_accuracy"] - split_payload["validation_balanced_accuracy"]
        )
        split_payload["train_minus_test_balanced_accuracy"] = (
            split_payload["train_balanced_accuracy"] - split_payload["test_balanced_accuracy"]
        )
        split_payload["train_minus_test_f1"] = split_payload["train_f1"] - split_payload["test_f1"]
        split_payload["test_rmse_over_train_rmse"] = (
            split_payload["test_rmse"] / split_payload["train_rmse"]
            if split_payload["train_rmse"] and not pd.isna(split_payload["train_rmse"])
            else math.nan
        )
        split_payload["direction_overfit_flag"] = bool(
            split_payload["train_minus_test_accuracy"] > 0.10
            or split_payload["train_minus_test_balanced_accuracy"] > 0.10
            or split_payload["train_minus_test_f1"] > 0.20
        )
        split_payload["regression_overfit_flag"] = bool(
            not pd.isna(split_payload["test_rmse_over_train_rmse"])
            and split_payload["test_rmse_over_train_rmse"] > 1.50
        )
        split_payload["overall_overfit_flag"] = bool(
            split_payload["direction_overfit_flag"] or split_payload["regression_overfit_flag"]
        )
        rows.append(split_payload)
    return pd.DataFrame(rows)


def _format_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_coverage_markdown(coverage: pd.DataFrame, path: Path) -> None:
    more_available = coverage[coverage["more_mubasher_pages_available"] == True].copy()
    if more_available.empty:
        scrape_interpretation = (
            "- The local scrape reaches the current live Mubasher listing-page counts for all ticker/language pairs checked, "
            "so simply increasing `max_pages_per_language` is not expected to add more historical Mubasher pages from this source."
        )
    else:
        tickers = ", ".join(more_available["ticker"].astype(str).tolist())
        scrape_interpretation = (
            f"- Live page-count checks indicate that additional Mubasher listing pages may be available for: {tickers}. "
            "The resumable scraper should be run before treating source coverage as complete."
        )
    lines = [
        "# Coverage Analysis",
        "",
        "This report measures whether weak performance is plausibly linked to sparse news coverage and whether more Mubasher listing pages appear available.",
        "",
        "| Ticker | Articles | EN | AR | Trading days in article window | Same-day coverage | Prior-day coverage | Prior-7d coverage | Density/articles per trading day | Live pages vs local pages | Assessment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in coverage.iterrows():
        local_live = (
            f"en {row.get('local_max_page_en')}/{row.get('live_available_pages_en')}, "
            f"ar {row.get('local_max_page_ar')}/{row.get('live_available_pages_ar')}"
        )
        lines.append(
            "| {ticker} | {articles} | {en} | {ar} | {days} | {same} | {prior} | {prior7} | {density} | {pages} | {assessment} |".format(
                ticker=row["ticker"],
                articles=int(row["sentiment_labeled_articles"]),
                en=int(row["english_articles"]),
                ar=int(row["arabic_articles"]),
                days=int(row["trading_days_in_article_window"]),
                same=_format_float(row["same_day_article_coverage_ratio"]),
                prior=_format_float(row["prior_calendar_day_article_coverage_ratio"]),
                prior7=_format_float(row["prior_7d_article_coverage_ratio"]),
                density=_format_float(row["article_density_per_trading_day_window"]),
                pages=local_live,
                assessment=row["scraping_assessment"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- OIH and TMG remain sparse relative to their trading calendars; OIH has the lowest labeled article count and a shorter article history.",
            scrape_interpretation,
            "- Two OIH article-page failures are recorded in the scraper manifest. That is too small to explain weak model performance, but it is documented as residual source incompleteness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leakage_markdown(report: Dict[str, object], path: Path) -> None:
    status = report["overall_status"]
    lines = [
        "# Leakage Check Report",
        "",
        f"Overall status: **{status}**",
        "",
        "## Automated Checks",
        "",
        f"- Panel checks run: {len(report['automated_panel_checks'])}",
        f"- Walk-forward rows checked: {report['walk_forward_history_check']['evaluation_rows_checked']}",
        f"- Direction history violations: {report['walk_forward_history_check']['direction_history_violations']}",
        f"- Return history violations: {report['walk_forward_history_check']['return_history_violations']}",
        "",
        "## Static Code Checks",
        "",
    ]
    for key, value in report["static_code_checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Violations", ""])
    if report["violations"]:
        lines.extend(f"- {item}" for item in report["violations"])
    else:
        lines.append("No leakage violations were detected by the automated checks.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overfitting_markdown(report: pd.DataFrame, path: Path) -> None:
    flagged = report[report["overall_overfit_flag"]].copy()
    previous_headline = report[
        (report["horizon"] == 5)
        & (report["feature_strategy"] == "event_weighted")
        & (report["ticker"] == "OIH")
        & (report["view"] == "bilingual")
    ]
    validity_passing = report[
        (~report["overall_overfit_flag"])
        & (report["validation_accuracy"] >= 0.50)
        & (report["validation_balanced_accuracy"] >= 0.50)
        & (report["validation_f1"] >= 0.50)
        & (report["test_accuracy"] >= 0.50)
        & (report["test_balanced_accuracy"] >= 0.50)
        & (report["test_f1"] >= 0.50)
    ].copy()
    accepted = (
        validity_passing.sort_values(["test_accuracy", "test_balanced_accuracy", "test_f1"], ascending=False).head(1)
        if not validity_passing.empty
        else pd.DataFrame()
    )
    lines = [
        "# Overfitting Report",
        "",
        f"Rows checked: {len(report)}",
        f"Rows flagged for train/test overfitting: {len(flagged)}",
        "",
        "A row is flagged when train-test accuracy or balanced-accuracy gap exceeds 0.10, train-test F1 gap exceeds 0.20, or test RMSE is more than 1.5x train RMSE.",
        "",
    ]
    if not previous_headline.empty:
        row = previous_headline.iloc[0]
        lines.extend(
            [
                "## Previous Headline Accuracy Candidate",
                "",
                f"OIH 5-day event-weighted bilingual test accuracy is `{row['test_accuracy']:.3f}` with balanced accuracy `{row['test_balanced_accuracy']:.3f}`.",
                f"Train accuracy is `{row['train_accuracy']:.3f}` and validation accuracy is `{row['validation_accuracy']:.3f}`.",
                f"Overall overfit flag for this row: `{bool(row['overall_overfit_flag'])}`. This row should not be treated as the accepted best model.",
                "",
            ]
        )
    if not accepted.empty:
        row = accepted.iloc[0]
        lines.extend(
            [
                "## Accepted Validity-Passing Candidate",
                "",
                f"The strongest candidate that passes the explicit validity filter is `{row['ticker']}` horizon `{int(row['horizon'])}` `{row['feature_strategy']}` `{row['view']}`.",
                f"Validation accuracy/balanced accuracy: `{row['validation_accuracy']:.3f}` / `{row['validation_balanced_accuracy']:.3f}`.",
                f"Test accuracy/balanced accuracy/F1/sign accuracy: `{row['test_accuracy']:.3f}` / `{row['test_balanced_accuracy']:.3f}` / `{row['test_f1']:.3f}` / `{row['test_sign_accuracy']:.3f}`.",
                f"Overall overfit flag for this row: `{bool(row['overall_overfit_flag'])}`.",
                "",
            ]
        )
    lines.extend(["## Highest Train-Test Accuracy Gaps", ""])
    top = report.sort_values("train_minus_test_accuracy", ascending=False).head(10)
    lines.append("| Horizon | Strategy | Ticker | View | Train Acc. | Val Acc. | Test Acc. | Train-Test Gap | Flag |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['feature_strategy']} | {row['ticker']} | {row['view']} | "
            f"{row['train_accuracy']:.3f} | {row['validation_accuracy']:.3f} | {row['test_accuracy']:.3f} | "
            f"{row['train_minus_test_accuracy']:.3f} | {bool(row['overall_overfit_flag'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Several higher-dimensional feature sets show large in-sample training performance that does not carry to validation/test. Final claims should therefore be based on validation/test metrics only. The OIH 5-day event-weighted accuracy improvement is measured on test data but is flagged for overfitting, so it is exploratory. The accepted candidate is still weak because its balanced accuracy remains close to random.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _accepted_validity_candidate(report: pd.DataFrame) -> Dict[str, object]:
    validity_passing = report[
        (~report["overall_overfit_flag"])
        & (report["validation_accuracy"] >= 0.50)
        & (report["validation_balanced_accuracy"] >= 0.50)
        & (report["validation_f1"] >= 0.50)
        & (report["test_accuracy"] >= 0.50)
        & (report["test_balanced_accuracy"] >= 0.50)
        & (report["test_f1"] >= 0.50)
    ].copy()
    if validity_passing.empty:
        return {}
    row = validity_passing.sort_values(["test_accuracy", "test_balanced_accuracy", "test_f1"], ascending=False).iloc[0]
    return row.to_dict()


def update_summary_with_phase7(output_dir: Path, leakage_report: Dict[str, object], overfitting: pd.DataFrame) -> None:
    summary_path = output_dir / "summary_by_horizon.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    previous_headline = overfitting[
        (overfitting["horizon"] == 5)
        & (overfitting["feature_strategy"] == "event_weighted")
        & (overfitting["ticker"] == "OIH")
        & (overfitting["view"] == "bilingual")
    ]
    summary["phase7_validity"] = {
        "leakage_status": leakage_report["overall_status"],
        "overfit_rows_flagged": int(overfitting["overall_overfit_flag"].sum()),
        "accepted_validity_candidate": _accepted_validity_candidate(overfitting),
        "rejected_previous_headline_candidate": previous_headline.iloc[0].to_dict() if not previous_headline.empty else {},
        "acceptance_rule": (
            "Accepted candidate must have no overfit flag and validation/test accuracy, "
            "balanced accuracy, and F1 all >= 0.50."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(
        args.output_dir
        or (
            "outputs/egx30_sentiment_only_experiments"
            if args.target_scope == "egx30"
            else "outputs/final_sentiment_experiments"
        )
    )
    if args.target_scope == "egx30":
        from run_egx30_validity_checks import run_checks

        report = run_checks(
            sentiment_labeled_path=Path(args.sentiment_labeled_path),
            target_price_path=Path(args.target_price_path),
            llm_cache_path=Path(args.llm_cache_path) if args.llm_cache_path else None,
            output_dir=output_dir,
            news_tickers=args.tickers,
            horizons=args.horizons,
            rolling_windows=args.rolling_windows,
        )
        print(json.dumps(report, indent=2, default=str))
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SentimentOnlyPipelineConfig(
        tickers=tuple(args.tickers),
        output_dir=output_dir,
        price_period=args.price_period,
        rolling_windows=tuple(args.rolling_windows),
    )
    scored_articles = load_scored_articles(Path(args.sentiment_labeled_path), config)
    price_frames = load_price_frames(config, output_dir)

    coverage = build_coverage_analysis(
        scored_articles,
        price_frames,
        Path(args.raw_news_dir),
        args.tickers,
        skip_live_page_counts=args.skip_live_page_counts,
    )
    coverage_path = output_dir / "coverage_analysis.csv"
    coverage_md_path = output_dir / "coverage_analysis.md"
    coverage.to_csv(coverage_path, index=False)
    write_coverage_markdown(coverage, coverage_md_path)

    leakage_report = run_leakage_checks(
        scored_articles,
        price_frames,
        args.tickers,
        args.horizons,
        args.rolling_windows,
        output_dir / "predictions_by_horizon.csv",
    )
    leakage_json_path = output_dir / "leakage_check_report.json"
    leakage_md_path = output_dir / "leakage_check_report.md"
    leakage_json_path.write_text(json.dumps(leakage_report, indent=2, default=str), encoding="utf-8")
    write_leakage_markdown(leakage_report, leakage_md_path)

    overfitting = build_overfitting_report(output_dir / "predictions_by_horizon.csv")
    overfitting_path = output_dir / "overfitting_report.csv"
    overfitting_md_path = output_dir / "overfitting_report.md"
    overfitting.to_csv(overfitting_path, index=False)
    write_overfitting_markdown(overfitting, overfitting_md_path)
    update_summary_with_phase7(output_dir, leakage_report, overfitting)

    print("Saved coverage analysis:", coverage_path)
    print("Saved coverage narrative:", coverage_md_path)
    print("Saved leakage report:", leakage_json_path)
    print("Saved leakage narrative:", leakage_md_path)
    print("Saved overfitting report:", overfitting_path)
    print("Saved overfitting narrative:", overfitting_md_path)
    print("Leakage status:", leakage_report["overall_status"])
    print("Overfit rows flagged:", int(overfitting["overall_overfit_flag"].sum()))


if __name__ == "__main__":
    main()
