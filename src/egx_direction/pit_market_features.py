"""Point-in-time full-universe EGX30 sentiment feature construction."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .market_index import EGX30_TICKER
from .pit_constituents import EGX30_RESEARCH_END, EGX30_RESEARCH_START, active_tickers


def _article_identity(row: pd.Series) -> str:
    article_id = str(row.get("article_id", "") or "").strip()
    url = str(row.get("article_url", row.get("url", "")) or "").strip()
    language = str(row.get("detected_language", row.get("article_language", "")) or "").strip().lower()
    identity = article_id or url
    if not identity:
        identity = str(row.get("clean_article_text", "") or "")
    return hashlib.sha1(f"{language}|{identity}".encode("utf-8")).hexdigest()


def scope_articles_to_pit_market(articles: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    """Keep active-company news and deduplicate stories attached to multiple pages."""

    if articles.empty:
        return articles.copy()
    frame = articles.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["Date", "Ticker"])
    frame = frame[
        frame["Date"].between(pd.Timestamp(EGX30_RESEARCH_START), pd.Timestamp(EGX30_RESEARCH_END), inclusive="both")
    ].copy()
    active_by_date = {
        date: set(active_tickers(constituents, date))
        for date in frame["Date"].drop_duplicates().tolist()
    }
    frame = frame[
        frame.apply(lambda row: row["Ticker"] in active_by_date.get(row["Date"], set()), axis=1)
    ].copy()
    frame["market_article_identity"] = frame.apply(_article_identity, axis=1)
    frame["sentiment_article_weight"] = pd.to_numeric(
        frame.get("sentiment_article_weight", 0.0), errors="coerce"
    ).fillna(0.0)
    frame = frame.sort_values(
        ["Date", "market_article_identity", "sentiment_article_weight", "Ticker"],
        ascending=[True, True, False, True],
    )
    source_tickers = (
        frame.groupby(["Date", "market_article_identity"])["Ticker"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("Source_Tickers")
    )
    frame = frame.drop_duplicates(subset=["Date", "market_article_identity"], keep="first")
    frame = frame.merge(source_tickers, on=["Date", "market_article_identity"], how="left")
    return frame.sort_values(["Date", "Ticker", "market_article_identity"]).reset_index(drop=True)


def _is_additive(column: str) -> bool:
    return (
        column.endswith("_count")
        or column.endswith("_total")
        or column.endswith("_sum")
        or "article_count" in column
        or column.endswith("_volume")
    )


def aggregate_equal_weight_market_daily(
    company_daily: pd.DataFrame,
    constituents: pd.DataFrame,
    *,
    start_date: str = EGX30_RESEARCH_START,
    end_date: str = EGX30_RESEARCH_END,
) -> pd.DataFrame:
    """Average firm-level mood equally across the active point-in-time EGX30 universe."""

    daily = company_daily.copy()
    if not daily.empty:
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
        daily["Ticker"] = daily["Ticker"].astype(str).str.upper()
    feature_columns = [column for column in daily.columns if column not in {"Ticker", "Date"}]
    rows: List[Dict[str, float]] = []

    for date in pd.date_range(start_date, end_date, freq="D"):
        active = list(active_tickers(constituents, date))
        active_count = len(active)
        date_frame = daily[daily["Date"] == date].copy() if not daily.empty else pd.DataFrame()
        if not date_frame.empty:
            date_frame = date_frame[date_frame["Ticker"].isin(active)].set_index("Ticker")
        else:
            date_frame = pd.DataFrame(index=pd.Index([], name="Ticker"))
        company_frame = date_frame.reindex(active)
        numeric = company_frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        row: Dict[str, float] = {"Ticker": EGX30_TICKER, "Date": date}
        for column in feature_columns:
            values = numeric[column] if column in numeric else pd.Series(0.0, index=active)
            row[column] = float(values.sum() if _is_additive(column) else values.mean())

        if "combined_article_count" in numeric:
            covered = int((numeric["combined_article_count"] > 0).sum())
        else:
            count_columns = [column for column in feature_columns if "article_count" in column]
            covered = int((numeric[count_columns].sum(axis=1) > 0).sum()) if count_columns else 0
        row["active_constituent_count"] = float(active_count)
        row["covered_constituent_count"] = float(covered)
        row["constituent_coverage_ratio"] = float(covered / active_count) if active_count else 0.0
        rows.append(row)

    result = pd.DataFrame(rows)
    numeric_columns = [column for column in result.columns if column not in {"Ticker", "Date"}]
    result[numeric_columns] = result[numeric_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return result.sort_values("Date").reset_index(drop=True)
