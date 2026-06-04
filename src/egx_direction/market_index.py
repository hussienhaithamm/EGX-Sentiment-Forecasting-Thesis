"""Target-only EGX30 index price ingestion for sentiment forecasting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


EGX30_TICKER = "EGX30"
EGX30_START_DATE = "2016-04-21"
EGX30_END_DATE = "2026-04-20"
EGX_INDEX_ENDPOINT = "https://www.egx.com.eg/WebService.asmx/getIndexChartData"
MUBASHER_EGX30_HISTORICAL_CSV = (
    "https://static.mubasher.info/File.MubasherCharts/"
    "File.Historical_Index_Charts_Dir/39d67d8ab4f4c10bf22aa353e27879133c.csv"
)
DEFAULT_EGX30_PRICE_PATH = Path("data/EGX30 Index Historical Data.csv")


def normalize_egx30_payload(
    payload: object,
    *,
    start_date: str = EGX30_START_DATE,
    end_date: str = EGX30_END_DATE,
) -> pd.DataFrame:
    """Normalize the official EGX index response into a target-only close frame."""

    if not isinstance(payload, list):
        raise ValueError("Official EGX index payload must be a JSON list.")

    frame = pd.DataFrame(payload)
    required = {"CDAY", "INDEX_VALUE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Official EGX index payload is missing columns: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["CDAY"], errors="coerce").dt.normalize(),
            "Close": pd.to_numeric(frame["INDEX_VALUE"].astype(str).str.replace(",", "", regex=False), errors="coerce"),
        }
    )
    normalized = normalized.dropna(subset=["Date", "Close"])
    normalized = normalized[
        normalized["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    ]
    normalized = normalized.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    normalized["Ticker"] = EGX30_TICKER
    return validate_egx30_prices(normalized, start_date=start_date, end_date=end_date)


def validate_egx30_prices(
    frame: pd.DataFrame,
    *,
    start_date: str = EGX30_START_DATE,
    end_date: str = EGX30_END_DATE,
    require_exact_bounds: bool = False,
) -> pd.DataFrame:
    """Validate a local EGX30 close series before it is used to create labels."""

    required = {"Date", "Close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EGX30 price frame is missing columns: {sorted(missing)}")

    clean = frame.copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce").dt.normalize()
    clean["Close"] = pd.to_numeric(clean["Close"], errors="coerce")
    clean = clean.dropna(subset=["Date", "Close"])
    clean = clean[
        clean["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    ]
    clean = clean.sort_values("Date").reset_index(drop=True)
    clean["Ticker"] = EGX30_TICKER

    if clean.empty:
        raise ValueError("EGX30 price frame is empty after timeline filtering.")
    if clean["Date"].duplicated().any():
        raise ValueError("EGX30 price frame contains duplicate trading dates.")
    if (clean["Close"] <= 0).any():
        raise ValueError("EGX30 price frame contains non-positive close values.")
    if require_exact_bounds:
        observed_start = clean["Date"].min().date().isoformat()
        observed_end = clean["Date"].max().date().isoformat()
        if observed_start != start_date or observed_end != end_date:
            raise ValueError(
                f"EGX30 price frame bounds are {observed_start} to {observed_end}; "
                f"expected {start_date} to {end_date}."
            )
    return clean[["Date", "Close", "Ticker"]]


def fetch_egx30_prices(
    *,
    start_date: str = EGX30_START_DATE,
    end_date: str = EGX30_END_DATE,
    timeout_seconds: float = 30.0,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch EGX30 closes from the official Egyptian Exchange web service."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    period_days = int((end - start).days) + 30
    client = session or requests.Session()
    response = client.get(
        EGX_INDEX_ENDPOINT,
        params={"index": EGX30_TICKER, "period": period_days, "gtk": 0},
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.egx.com.eg/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return normalize_egx30_payload(response.json(), start_date=start_date, end_date=end_date)


def fetch_mubasher_egx30_prices(
    *,
    start_date: str = EGX30_START_DATE,
    end_date: str = EGX30_END_DATE,
    timeout_seconds: float = 30.0,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch the public Mubasher EGX30 historical chart CSV."""

    client = session or requests.Session()
    response = client.get(
        MUBASHER_EGX30_HISTORICAL_CSV,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://english.mubasher.info/markets/EGX/indices/EGX30/",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    frame = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=["Date", "Open", "High", "Low", "Close", "Volume"],
    )
    frame["Date"] = frame["Date"].astype(str).str.replace("/00:00:00", "", regex=False)
    frame = frame.drop_duplicates(subset=["Date"], keep="last")
    return validate_egx30_prices(frame, start_date=start_date, end_date=end_date)


def load_egx30_prices(
    path: Path = DEFAULT_EGX30_PRICE_PATH,
    *,
    start_date: str = EGX30_START_DATE,
    end_date: str = EGX30_END_DATE,
    require_exact_bounds: bool = True,
) -> pd.DataFrame:
    """Load a pinned local EGX30 close series. Live fallback is intentionally absent."""

    if not path.exists():
        raise FileNotFoundError(
            f"Pinned EGX30 close file is missing: {path}. "
            "Run scripts/fetch_egx30_prices.py when the official EGX endpoint is reachable."
        )
    return validate_egx30_prices(
        pd.read_csv(path),
        start_date=start_date,
        end_date=end_date,
        require_exact_bounds=require_exact_bounds,
    )


def write_egx30_artifacts(
    frame: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_EGX30_PRICE_PATH,
    source_url: str = EGX_INDEX_ENDPOINT,
) -> Dict[str, object]:
    """Persist the pinned target series and an auditable metadata sidecar."""

    clean = validate_egx30_prices(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "target": EGX30_TICKER,
        "usage": "target_construction_only",
        "rows": int(len(clean)),
        "first_date": clean["Date"].min().date().isoformat(),
        "last_date": clean["Date"].max().date().isoformat(),
        "sha256": digest,
        "csv_path": str(output_path),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
