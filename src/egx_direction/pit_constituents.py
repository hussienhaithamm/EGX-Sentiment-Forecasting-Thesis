"""Point-in-time EGX30 constituent registry helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

from news_pipeline.tickers import TargetCompany


EGX30_RESEARCH_START = "2021-04-21"
EGX30_RESEARCH_END = "2026-04-20"
DEFAULT_CONSTITUENT_PATH = Path("data/egx30_historical_constituents_5yr.csv")
DEFAULT_COMPANY_PATH = Path("data/egx30_mubasher_company_registry.csv")
REQUIRED_CONSTITUENT_COLUMNS = {
    "Date_From",
    "Date_To",
    "Ticker",
    "Mubasher_Slug",
    "Company_Name",
}


@dataclass(frozen=True)
class TransitionAllowance:
    """Documented non-30 membership state caused by a corporate action."""

    date_from: str
    date_to: str
    expected_company_count: int
    reason: str


TRANSITION_ALLOWANCES: Mapping[Tuple[str, str], TransitionAllowance] = {
    ("2021-04-21", "2021-07-31"): TransitionAllowance(
        date_from="2021-04-21",
        date_to="2021-07-31",
        expected_company_count=31,
        reason="Orascom Investment Holding demerger transition: OIH and OFH were both represented.",
    ),
    ("2021-10-13", "2022-01-31"): TransitionAllowance(
        date_from="2021-10-13",
        date_to="2022-01-31",
        expected_company_count=32,
        reason="Pioneers Holding demerger transition: the three successor securities temporarily entered EGX30.",
    ),
}


def load_constituent_registry(path: Path = DEFAULT_CONSTITUENT_PATH) -> pd.DataFrame:
    """Load and normalize the historical company-level membership table."""

    frame = pd.read_csv(path)
    missing = REQUIRED_CONSTITUENT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Constituent registry is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Date_From"] = pd.to_datetime(frame["Date_From"], errors="raise").dt.normalize()
    frame["Date_To"] = pd.to_datetime(frame["Date_To"], errors="raise").dt.normalize()
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
    frame["Mubasher_Slug"] = frame["Mubasher_Slug"].astype(str).str.strip().str.upper()
    return frame.sort_values(["Date_From", "Date_To", "Ticker"]).reset_index(drop=True)


def validate_constituent_registry(
    frame: pd.DataFrame,
    *,
    start_date: str = EGX30_RESEARCH_START,
    end_date: str = EGX30_RESEARCH_END,
) -> Dict[str, object]:
    """Assert continuous coverage and only documented non-30 transition states."""

    registry = frame.copy()
    missing = REQUIRED_CONSTITUENT_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"Constituent registry is missing columns: {sorted(missing)}")
    registry["Date_From"] = pd.to_datetime(registry["Date_From"], errors="raise").dt.normalize()
    registry["Date_To"] = pd.to_datetime(registry["Date_To"], errors="raise").dt.normalize()
    if (registry["Date_To"] < registry["Date_From"]).any():
        raise ValueError("Constituent registry contains an interval with Date_To before Date_From.")
    if registry.duplicated(subset=["Date_From", "Date_To", "Ticker"]).any():
        raise ValueError("Constituent registry contains duplicate company rows.")

    intervals = []
    for (date_from, date_to), group in registry.groupby(["Date_From", "Date_To"], sort=True):
        interval_key = (date_from.date().isoformat(), date_to.date().isoformat())
        count = int(group["Ticker"].nunique())
        allowance = TRANSITION_ALLOWANCES.get(interval_key)
        expected = allowance.expected_company_count if allowance else 30
        if count != expected:
            raise ValueError(
                f"EGX30 interval {interval_key[0]} to {interval_key[1]} has {count} companies; expected {expected}."
            )
        intervals.append(
            {
                "date_from": interval_key[0],
                "date_to": interval_key[1],
                "company_count": count,
                "transition_allowance": asdict(allowance) if allowance else None,
            }
        )

    if not intervals:
        raise ValueError("Constituent registry is empty.")
    if intervals[0]["date_from"] != start_date or intervals[-1]["date_to"] != end_date:
        raise ValueError(
            f"Constituent registry bounds are {intervals[0]['date_from']} to {intervals[-1]['date_to']}; "
            f"expected {start_date} to {end_date}."
        )
    for previous, current in zip(intervals, intervals[1:]):
        expected_next = (pd.Timestamp(previous["date_to"]) + pd.Timedelta(days=1)).date().isoformat()
        if current["date_from"] != expected_next:
            raise ValueError(
                f"Constituent registry gap or overlap after {previous['date_to']}: next starts {current['date_from']}."
            )

    return {
        "status": "PASS",
        "start_date": start_date,
        "end_date": end_date,
        "interval_count": len(intervals),
        "unique_company_count": int(registry["Ticker"].nunique()),
        "intervals": intervals,
    }


def active_tickers(frame: pd.DataFrame, date: object) -> Tuple[str, ...]:
    """Return company identities active in EGX30 on one calendar date."""

    value = pd.Timestamp(date).normalize()
    date_from = pd.to_datetime(frame["Date_From"], errors="raise").dt.normalize()
    date_to = pd.to_datetime(frame["Date_To"], errors="raise").dt.normalize()
    active = frame[(date_from <= value) & (date_to >= value)]
    return tuple(sorted(active["Ticker"].dropna().astype(str).str.upper().unique()))


def load_mubasher_companies(path: Path = DEFAULT_COMPANY_PATH) -> Dict[str, TargetCompany]:
    """Load the dynamic company union used by the full-universe scraper."""

    frame = pd.read_csv(path).fillna("")
    required = {"Ticker", "Mubasher_Slug", "Company_Name", "Company_Name_Ar", "Aliases"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Company registry is missing columns: {sorted(missing)}")

    companies: Dict[str, TargetCompany] = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row["Ticker"]).strip().upper()
        aliases = tuple(value.strip() for value in str(row["Aliases"]).split("|") if value.strip())
        companies[ticker] = TargetCompany(
            ticker=ticker,
            site_code=str(row["Mubasher_Slug"]).strip().upper(),
            company_name_en=str(row["Company_Name"]).strip(),
            company_name_ar=str(row["Company_Name_Ar"]).strip(),
            primary_aliases=aliases,
        )
    return companies
