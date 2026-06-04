"""Market-mood feature preparation for an EGX30 sentiment-only target."""

from __future__ import annotations

from typing import List

import pandas as pd

from egx_direction.market_index import EGX30_TICKER


SOURCE_TICKERS = ("CIB", "TMG", "OIH")
MARKET_SCOPE_FEATURE_COLUMNS = (
    "source_company_count",
    "source_cib_article_count",
    "source_tmg_article_count",
    "source_oih_article_count",
    "firm_article_count",
    "macro_article_count",
)


def _deduplication_key(row: pd.Series) -> str:
    language = str(row.get("detected_language", row.get("article_language", "")) or "")
    article_id = str(row.get("article_id", "") or "").strip()
    article_url = str(row.get("article_url", row.get("url", "")) or "").strip()
    identity = article_id or article_url
    if not identity:
        identity = "|".join(
            [
                str(row.get("Date", "") or ""),
                str(row.get("cleaned_title", row.get("title", "")) or ""),
            ]
        )
    return f"{language}|{identity}"


def scope_articles_to_market(
    scored_articles: pd.DataFrame,
    *,
    target_ticker: str = EGX30_TICKER,
) -> pd.DataFrame:
    """Pool firm and macro articles into one deduplicated EGX30 news stream."""

    if scored_articles.empty:
        return scored_articles.copy()

    frame = scored_articles.copy()
    frame["SourceTicker"] = frame.get("SourceTicker", frame.get("Ticker", "")).fillna("").astype(str)
    if "NewsScope" not in frame.columns:
        frame["NewsScope"] = "firm"
    frame["NewsScope"] = frame["NewsScope"].fillna("firm").astype(str).str.lower()
    frame["_market_dedup_key"] = frame.apply(_deduplication_key, axis=1)
    if "sentiment_article_weight" in frame.columns:
        market_weight = frame["sentiment_article_weight"]
    else:
        market_weight = pd.Series(0.0, index=frame.index)
    frame["_market_weight"] = pd.to_numeric(market_weight, errors="coerce").fillna(0.0)

    rows: List[pd.Series] = []
    for _, group in frame.groupby("_market_dedup_key", sort=True):
        selected = group.sort_values("_market_weight", ascending=False).iloc[0].copy()
        source_tickers = sorted({value for value in group["SourceTicker"].astype(str) if value})
        selected["SourceTickers"] = "|".join(source_tickers)
        selected["SourceTicker"] = source_tickers[0] if len(source_tickers) == 1 else "MULTI"
        selected["source_company_count"] = float(len({value for value in source_tickers if value in SOURCE_TICKERS}))
        selected["Ticker"] = target_ticker
        rows.append(selected)

    scoped = pd.DataFrame(rows).drop(columns=["_market_dedup_key", "_market_weight"], errors="ignore")
    return scoped.sort_values(["Date", "published_at", "article_id"]).reset_index(drop=True)


def aggregate_daily_market_scope_features(scoped_articles: pd.DataFrame, view: str) -> pd.DataFrame:
    """Count market-news sources per day without using price information."""

    if scoped_articles.empty:
        return pd.DataFrame(columns=["Ticker", "Date", *MARKET_SCOPE_FEATURE_COLUMNS])
    if view == "english_only":
        frame = scoped_articles[scoped_articles["detected_language"] == "en"].copy()
    elif view == "arabic_only":
        frame = scoped_articles[scoped_articles["detected_language"] == "ar"].copy()
    elif view == "bilingual":
        frame = scoped_articles[scoped_articles["detected_language"].isin({"en", "ar"})].copy()
    else:
        raise ValueError(f"Unsupported view: {view}")

    rows = []
    for (ticker, date), group in frame.groupby(["Ticker", "Date"], sort=True):
        source_sets = group["SourceTickers"].fillna("").astype(str).map(
            lambda value: {item for item in value.split("|") if item}
        )
        all_sources = set().union(*source_sets.tolist()) if len(source_sets) else set()
        row = {
            "Ticker": ticker,
            "Date": date,
            "source_company_count": float(len(all_sources & set(SOURCE_TICKERS))),
            "firm_article_count": float(group["NewsScope"].eq("firm").sum()),
            "macro_article_count": float(group["NewsScope"].eq("macro").sum()),
        }
        for source_ticker in SOURCE_TICKERS:
            row[f"source_{source_ticker.lower()}_article_count"] = float(
                sum(source_ticker in sources for sources in source_sets)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)
