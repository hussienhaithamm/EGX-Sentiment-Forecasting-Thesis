"""Analyze association between daily news sentiment and subsequent stock returns.

The script is intentionally descriptive: it estimates whether stronger positive
or negative sentiment days are associated with future stock returns, without
claiming causality from correlation alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PRICE_FILES = {
    "CIB": "Commercial Int Bank Stock Price History",
    "TMG": "TMG Holding Stock Price History",
    "OIH": "Orascom Investment Holding",
}

SENTIMENT_COLUMNS = (
    "combined_sentiment_weighted_mean",
    "combined_sentiment_weighted_volume",
    "combined_sentiment_polarity_share",
    "combined_article_count",
    "combined_scored_count",
)
SENTIMENT_LAG_CALENDAR_DAYS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate news sentiment with EGX stock returns.")
    parser.add_argument(
        "--daily-sentiment-path",
        default="outputs/final_sentiment_experiments/features/h5_all_news_bilingual_daily_features.csv",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/sentiment_stock_impact_analysis")
    parser.add_argument("--tickers", nargs="+", default=["CIB", "TMG", "OIH"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[0, 1, 3, 5])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=5)
    return parser.parse_args()


def load_price_frame(data_dir: Path, ticker: str) -> pd.DataFrame:
    matches = sorted(data_dir.glob(f"{PRICE_FILES[ticker]}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No price CSV found for {ticker}")
    frame = pd.read_csv(matches[0])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    frame["Ticker"] = ticker
    frame["daily_return"] = frame["Close"].pct_change()
    for horizon in (1, 3, 5):
        frame[f"future_return_{horizon}d"] = frame["Close"].shift(-horizon) / frame["Close"] - 1.0
    frame["future_return_0d"] = frame["daily_return"]
    return frame


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "spearman":
        x = _rank(x)
        y = _rank(y)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _block_permutation(values: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    n = len(values)
    if n <= block_size:
        return rng.permutation(values)
    blocks = [values[start : min(start + block_size, n)] for start in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[idx] for idx in order])[:n]


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    permutations: int,
    block_size: int,
    seed: int,
) -> float:
    observed = _corr(x, y, method)
    if math.isnan(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = _block_permutation(x, rng, block_size)
        candidate = _corr(shuffled, y, method)
        if not math.isnan(candidate) and abs(candidate) >= abs(observed):
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def build_analysis_panel(
    prices: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    ticker: str,
    horizons: Sequence[int],
) -> pd.DataFrame:
    ticker_daily = daily_sentiment[daily_sentiment["Ticker"] == ticker].copy()
    ticker_daily["Date"] = pd.to_datetime(ticker_daily["Date"], errors="coerce").dt.normalize()
    calendar_index = pd.date_range(
        min(prices["Date"].min(), ticker_daily["Date"].min()),
        prices["Date"].max(),
        freq="D",
    )
    for column in SENTIMENT_COLUMNS:
        if column not in ticker_daily.columns:
            ticker_daily[column] = 0.0
    daily = ticker_daily.set_index("Date")[list(SENTIMENT_COLUMNS)].reindex(calendar_index).fillna(0.0)

    shifted = daily.shift(SENTIMENT_LAG_CALENDAR_DAYS).fillna(0.0)
    aligned = shifted.loc[pd.Index(prices["Date"])].reset_index().rename(columns={"index": "Date"})
    aligned["Ticker"] = ticker
    aligned = aligned.rename(columns={column: f"lag1_{column}" for column in SENTIMENT_COLUMNS})

    panel = prices.merge(aligned, on=["Date", "Ticker"], how="left")
    for horizon in horizons:
        if f"future_return_{horizon}d" not in panel.columns:
            panel[f"future_return_{horizon}d"] = panel["Close"].shift(-horizon) / panel["Close"] - 1.0
    return panel


def correlation_rows(
    panel: pd.DataFrame,
    ticker: str,
    horizons: Sequence[int],
    permutations: int,
    block_size: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sentiment_features = [f"lag1_{column}" for column in SENTIMENT_COLUMNS]
    for feature in sentiment_features:
        for horizon in horizons:
            target = f"future_return_{horizon}d"
            subset = panel[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            subset = subset[subset[feature] != 0.0].copy()
            x = subset[feature].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)
            for method in ("pearson", "spearman"):
                corr = _corr(x, y, method)
                p_value = permutation_p_value(
                    x,
                    y,
                    method=method,
                    permutations=permutations,
                    block_size=block_size,
                    seed=42 + horizon + len(feature) + (0 if method == "pearson" else 10_000),
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "sentiment_lag_calendar_days": SENTIMENT_LAG_CALENDAR_DAYS,
                        "sentiment_feature": feature,
                        "return_horizon": horizon,
                        "correlation_method": method,
                        "rows": int(len(subset)),
                        "correlation": corr,
                        "permutation_p_value": p_value,
                    }
                )
    return rows


def event_study_rows(panel: pd.DataFrame, ticker: str) -> List[Dict[str, object]]:
    feature = "lag1_combined_sentiment_weighted_mean"
    article_count = "lag1_combined_article_count"
    sample = panel[[feature, article_count, "future_return_1d", "future_return_3d", "future_return_5d"]].copy()
    news_days = sample[sample[article_count] > 0].copy()
    rows: List[Dict[str, object]] = []
    if news_days.empty:
        return rows

    positive_cut = news_days[feature].quantile(0.75)
    negative_cut = news_days[feature].quantile(0.25)
    groups = {
        "top_sentiment_quartile": news_days[news_days[feature] >= positive_cut],
        "bottom_sentiment_quartile": news_days[news_days[feature] <= negative_cut],
        "all_news_days": news_days,
        "no_prior_news_days": sample[sample[article_count] == 0],
    }
    for label, group in groups.items():
        for horizon in (1, 3, 5):
            values = group[f"future_return_{horizon}d"].dropna()
            rows.append(
                {
                    "ticker": ticker,
                    "sentiment_lag_calendar_days": SENTIMENT_LAG_CALENDAR_DAYS,
                    "group": label,
                    "horizon": horizon,
                    "rows": int(len(values)),
                    "mean_return": float(values.mean()) if len(values) else float("nan"),
                    "median_return": float(values.median()) if len(values) else float("nan"),
                    "positive_return_share": float((values > 0).mean()) if len(values) else float("nan"),
                }
            )
    return rows


def write_report(correlations: pd.DataFrame, events: pd.DataFrame, output_path: Path) -> None:
    main = correlations[
        (correlations["sentiment_feature"] == "lag1_combined_sentiment_weighted_mean")
        & (correlations["correlation_method"] == "spearman")
    ].copy()
    strongest = main.reindex(main["correlation"].abs().sort_values(ascending=False).index).head(6)
    lines = [
        "# Sentiment Score and Stock Impact Analysis",
        "",
        "This experiment tests whether daily bilingual news sentiment is associated with same-day and future stock returns. Sentiment is shifted by one calendar day before being aligned to trading dates, so predictive correlations use only prior news.",
        "",
        "Permutation p-values shuffle sentiment in five-day blocks to reduce the chance that serial dependence alone creates an apparently significant correlation.",
        "",
        "## Main Spearman Correlations",
        "",
        "| Ticker | Return horizon | Rows | Correlation | Permutation p-value |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in strongest.iterrows():
        lines.append(
            f"| {row['ticker']} | {int(row['return_horizon'])} | {int(row['rows'])} | "
            f"{row['correlation']:.3f} | {row['permutation_p_value']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Event-Style Return Groups",
            "",
            "| Ticker | Group | Horizon | Rows | Mean return | Median return | Positive return share |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in events.sort_values(["ticker", "horizon", "group"]).iterrows():
        lines.append(
            f"| {row['ticker']} | {row['group']} | {int(row['horizon'])} | {int(row['rows'])} | "
            f"{row['mean_return']:.4f} | {row['median_return']:.4f} | {row['positive_return_share']:.3f} |"
        )

    significant = main[(main["permutation_p_value"] <= 0.05) & (main["correlation"].abs() >= 0.10)]
    lines.extend(["", "## Interpretation", ""])
    if significant.empty:
        lines.append(
            "No strong, stable monotonic sentiment-return association appears under the pre-specified threshold of absolute Spearman correlation >= 0.10 and permutation p-value <= 0.05."
        )
    else:
        tickers = ", ".join(sorted(significant["ticker"].unique()))
        lines.append(
            f"Some ticker-horizon pairs meet the exploratory association threshold, mainly for: {tickers}. These should be treated as descriptive associations rather than causal evidence."
        )
    lines.append(
        "The event-style grouping is useful for interpretation, but it remains sensitive to sparse coverage and market regime effects."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_sentiment = pd.read_csv(args.daily_sentiment_path)
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"], errors="coerce").dt.normalize()

    all_panels: List[pd.DataFrame] = []
    all_correlations: List[Dict[str, object]] = []
    all_events: List[Dict[str, object]] = []
    for ticker in args.tickers:
        prices = load_price_frame(Path(args.data_dir), ticker)
        panel = build_analysis_panel(prices, daily_sentiment, ticker, args.horizons)
        all_panels.append(panel)
        all_correlations.extend(correlation_rows(panel, ticker, args.horizons, args.permutations, args.block_size))
        all_events.extend(event_study_rows(panel, ticker))

    panels = pd.concat(all_panels, ignore_index=True)
    correlations = pd.DataFrame(all_correlations)
    events = pd.DataFrame(all_events)

    panels.to_csv(output_dir / "sentiment_return_panel.csv", index=False)
    correlations.to_csv(output_dir / "sentiment_return_correlations.csv", index=False)
    events.to_csv(output_dir / "sentiment_event_study.csv", index=False)
    write_report(correlations, events, output_dir / "sentiment_stock_impact_report.md")
    config = vars(args).copy()
    config["sentiment_lag_calendar_days"] = SENTIMENT_LAG_CALENDAR_DAYS
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print("Saved sentiment impact analysis to", output_dir)


if __name__ == "__main__":
    main()
