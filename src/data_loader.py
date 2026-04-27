"""
Stock data loader module
Provides StockDataLoader class to load multiple CSV files (Investing.com style and similar),
standardize columns, parse volumes, parse dates, sort by date and cache results in-memory.

This variant:
 - looks for four specific filenames inside a `data/` folder
 - if the files are missing it attempts to pull the last 4 years of history from Yahoo via yfinance
 - raises a clear error if neither local files nor yfinance data can be obtained

Usage (example):
    from data_loader import StockDataLoader

    loader = StockDataLoader(data_dir='data')
    loader.load_all()
    df = loader.get_raw('CIB')

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except Exception:
    _YFINANCE_AVAILABLE = False


# -- Helpers -----------------------------------------------------------------
_VOLUME_REPLACEMENTS = {
    "K": 1_000,
    "M": 1_000_000,
}

_LOCAL_FILE_STEMS_BY_TICKER = {
    "CIB": ("Commercial Int Bank Stock Price History",),
    "ORASCOM": ("Orascom Construction Ltd Stock Price History",),
    "BELTONE": ("Beltone Financial Hld Stock Price History",),
    "ELSEWEDY": ("Elsewedy Electric",),
    # Optional local aliases for the sentiment-only pipeline tickers.
    "TMG": ("TMG Holding Stock Price History", "TMG Holding", "Talaat Moustafa", "TMGH"),
    "OIH": ("Orascom Investment Holding", "OIH"),
}

_YFINANCE_SYMBOLS_BY_TICKER = {
    "CIB": ("COMI.CA", "EGS60121C018.CA"),
    "ORASCOM": ("ORAS.CA", "EGS95001C011.CA"),
    "BELTONE": ("BTFH.CA", "EGS691G1C015.CA"),
    "ELSEWEDY": ("SWDY.CA",),
    "TMG": ("TMGH.CA", "EGS691S1C011.CA"),
    "OIH": ("OIH.CA", "EGS693V1C014.CA"),
}

_TICKER_ALIASES = {
    "CIB": "CIB",
    "COMI": "CIB",
    "COMI.CA": "CIB",
    "ORASCOM": "ORASCOM",
    "ORAS": "ORASCOM",
    "ORAS.CA": "ORASCOM",
    "BELTONE": "BELTONE",
    "BTFH": "BELTONE",
    "BTFH.CA": "BELTONE",
    "ELSEWEDY": "ELSEWEDY",
    "SWDY": "ELSEWEDY",
    "SWDY.CA": "ELSEWEDY",
    "TMG": "TMG",
    "TMGH": "TMG",
    "TMGH.CA": "TMG",
    "TALAAT MOUSTAFA GROUP": "TMG",
    "TALAAT MOUSTAFA GROUP HOLDING": "TMG",
    "OIH": "OIH",
    "OIH.CA": "OIH",
    "ORASCOM INVESTMENT HOLDING": "OIH",
}


def _parse_volume_value(v: str) -> Optional[float]:
    """Parse human-readable volume strings ("1,234", "12.3K", "4M") to float.
    Returns None when parsing fails or value is NaN-like.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if s == "":
        return None

    # Remove commas and spaces
    s = s.replace(",", "").replace(" ", "")

    # Match trailing K/M (case-insensitive)
    m = re.match(r"^([0-9.+-]+)([KMkm])?$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return None

    num, suffix = m.groups()
    try:
        val = float(num)
    except Exception:
        return None

    if suffix:
        mult = _VOLUME_REPLACEMENTS.get(suffix.upper(), 1)
        return val * mult

    return val


def _standardize_columns(df: pd.DataFrame,
                         date_candidates: Sequence[str] = ("Date", "date", "DATA"),
                         close_candidates: Sequence[str] = ("Close", "Price", "Adj Close"),
                         volume_candidates: Sequence[str] = ("Volume", "Vol.", "Vol", "Trade Vol")) -> pd.DataFrame:
    """Rename columns in-place (returns a new DataFrame) to a normalized set: ['Date','Close','Volume'] when possible."""
    df = df.copy()
    cols = {c: c for c in df.columns}

    # Find date
    for cand in date_candidates:
        if cand in df.columns:
            cols[cand] = "Date"
            break

    # Find close/price
    for cand in close_candidates:
        if cand in df.columns:
            cols[cand] = "Close"
            break

    # Find volume
    for cand in volume_candidates:
        if cand in df.columns:
            cols[cand] = "Volume"
            break

    df = df.rename(columns=cols)
    return df


def _parse_percent_value(v: object) -> Optional[float]:
    """Parse percentage-like strings such as '-0.89%' into decimal floats."""

    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1]
    s = s.replace(",", "")
    try:
        return float(s) / 100.0
    except Exception:
        return None


# -- StockDataLoader --------------------------------------------------------
class StockDataLoader:
    """Loads and standardizes stock CSVs from `data_dir`.

    Behavior and defaults:
    - Looks for the following files (case-insensitive, any extension) inside `data_dir`:
        * "Commercial Int Bank Stock Price History"
        * "Orascom Construction Ltd Stock Price History"
        * "Beltone Financial Hld Stock Price History"
        * "Elsewedy Electric"
      If found, those files are loaded and standardized.

    - If any files are missing, the loader will attempt to download the last 4 years of history
      using yfinance for the corresponding tickers (EGX variants). If yfinance is not installed or
      download fails, a clear RuntimeError is raised.

    - The loader returns DataFrames with at least ['Date','Close'] and optionally 'Volume' and always a 'Ticker' column.

    """

    # filenames to look for (filename stem, expected ticker short name)
    _EXPECTED_FILES = [
        ("Commercial Int Bank Stock Price History", "CIB"),
        ("Orascom Construction Ltd Stock Price History", "ORASCOM"),
        ("Beltone Financial Hld Stock Price History", "BELTONE"),
        ("Elsewedy Electric", "ELSEWEDY"),
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.stocks: Dict[str, pd.DataFrame] = {}

    # --------------------- discovery / loading -------------------------------
    def _find_local_files(self) -> List[Tuple[Path, str]]:
        """Return list of (path, ticker) for expected files found in data_dir.
        Matching on stem contains expected name (case-insensitive), any extension allowed.
        """
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            return []

        found: List[Tuple[Path, str]] = []
        for expected_stem, ticker in self._EXPECTED_FILES:
            # search for any file whose stem contains the expected text (case-insensitive)
            matches = [p for p in self.data_dir.iterdir() if p.is_file() and expected_stem.lower() in p.stem.lower()]
            if matches:
                # if multiple matches pick the first (sorted by name)
                m = sorted(matches)[0]
                found.append((m, ticker))
        return found

    def _find_local_file_for_ticker(self, ticker: str) -> Optional[Path]:
        """Return the best matching local file for a supported ticker, if any."""

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            return None

        stems = list(_LOCAL_FILE_STEMS_BY_TICKER.get(ticker, ()))
        stems.extend([ticker, f"{ticker}_price_history"])
        for expected_stem in stems:
            matches = [
                path
                for path in self.data_dir.iterdir()
                if path.is_file() and expected_stem.lower() in path.stem.lower()
            ]
            if matches:
                return sorted(matches)[0]
        return None

    @staticmethod
    def _canonicalize_ticker(ticker: str) -> str:
        normalized = str(ticker).strip().upper()
        if normalized in _TICKER_ALIASES:
            return _TICKER_ALIASES[normalized]
        normalized_title = str(ticker).strip().upper()
        if normalized_title in _TICKER_ALIASES:
            return _TICKER_ALIASES[normalized_title]
        raise KeyError(f"Unsupported ticker: {ticker}")

    def _load_csv_like(self, p: Path, ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f" Error reading {p}: {e}")
            return None

        df = _standardize_columns(df)

        # Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Volume
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].apply(_parse_volume_value)

        # Close
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')

        for price_col in ['Open', 'High', 'Low']:
            if price_col in df.columns:
                df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(',', ''), errors='coerce')

        if 'Change %' in df.columns:
            df['Change %'] = df['Change %'].apply(_parse_percent_value)

        df['Ticker'] = ticker

        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)

        return df

    def _download_yfinance(self, ticker_short: str, period: str = "4y") -> Optional[pd.DataFrame]:
        if not _YFINANCE_AVAILABLE:
            print("  yfinance is not installed; cannot download fallback data.")
            return None

        yf_tickers = _YFINANCE_SYMBOLS_BY_TICKER.get(ticker_short)
        if not yf_tickers:
            print(f"  No yfinance fallback configured for {ticker_short}")
            return None

        for yf_ticker in yf_tickers:
            try:
                print(f"  → Downloading {ticker_short} data from yfinance (ticker={yf_ticker}) for last {period}...")
                data = yf.download(yf_ticker, period=period, progress=False, auto_adjust=False)
                if data is None or data.empty:
                    print(f"  ✗ yfinance returned no data for {yf_ticker}")
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                df = data.reset_index().rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })

                cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                df = df[cols]
                df['Ticker'] = ticker_short
                df = df.sort_values('Date').reset_index(drop=True)
                return df
            except Exception as e:
                print(f"  ✗ Exception while downloading {yf_ticker}: {e}")

        return None

    def load_all(self, allow_yfinance_fallback: bool = True) -> None:
        """Load the expected stocks.

        Steps:
         1. Try to locate and load expected files in data_dir.
         2. If any expected file is missing and allow_yfinance_fallback is True, attempt to download those tickers from yfinance for the last 4 years.
         3. If after both steps we still don't have all requested tickers, raise RuntimeError.
        """
        found = self._find_local_files()

        # Load local files first
        for path, ticker in found:
            df = self._load_csv_like(path, ticker)
            if df is not None:
                self.stocks[ticker] = df
                print(f"  Loaded local: {path.name} -> {ticker} ({len(df)} rows)")

        # Determine which tickers are still missing
        missing = [ticker for _, ticker in self._EXPECTED_FILES if ticker not in self.stocks]

        # Try yfinance fallback for missing ones
        if missing and allow_yfinance_fallback:
            for ticker in missing:
                df = self._download_yfinance(ticker, period='4y')
                if df is not None:
                    self.stocks[ticker] = df
                    print(f"  Downloaded fallback for {ticker} ({len(df)} rows)")

        # Final check
        missing_final = [ticker for _, ticker in self._EXPECTED_FILES if ticker not in self.stocks]
        if missing_final:
            raise RuntimeError(
                f"Could not obtain data for the following tickers: {missing_final}."
            )

    def load_ticker(self, ticker: str, allow_yfinance_fallback: bool = True, period: str = "4y") -> pd.DataFrame:
        """Load one ticker using the same local-file and yfinance logic as load_all()."""

        ticker = self._canonicalize_ticker(ticker)

        local_path = self._find_local_file_for_ticker(ticker)
        df = None
        if local_path is not None:
            df = self._load_csv_like(local_path, ticker)
            if df is not None:
                self.stocks[ticker] = df
                return df

        if allow_yfinance_fallback:
            df = self._download_yfinance(ticker, period=period)
            if df is not None:
                self.stocks[ticker] = df
                return df

        raise RuntimeError(f"Could not obtain data for ticker: {ticker}.")

    # --------------------- accessors / utilities ----------------------------
    def list_tickers(self) -> List[str]:
        return sorted(self.stocks.keys())

    def get_raw(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return the raw standardized DataFrame for ticker, or None if not loaded."""
        return self.stocks.get(ticker)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test of StockDataLoader")
    parser.add_argument("--data_dir", type=str, help="directory with data files (default 'data')", default='data')
    parser.add_argument("--no-yf", dest='yf', action='store_false', help='disable yfinance fallback')
    args = parser.parse_args()

    loader = StockDataLoader(data_dir=args.data_dir)
    try:
        loader.load_all(allow_yfinance_fallback=args.yf)
        print("Loaded tickers:", loader.list_tickers())
    except Exception as e:
        print("ERROR:", e)
