"""Loader utilities for structured news datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .storage import load_jsonl_records


class NewsDataLoader:
    """Load structured news JSONL files for downstream preprocessing."""

    def __init__(self, data_dir: str = "data/raw/news/al_mobasher"):
        self.data_dir = Path(data_dir)
        self.news: Dict[str, pd.DataFrame] = {}

    def _ticker_path(self, ticker: str) -> Path:
        return self.data_dir / f"{ticker}.jsonl"

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        path = self._ticker_path(ticker)
        records = load_jsonl_records(path)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(records)
        if "published_at" in frame.columns:
            frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        return frame

    def load_all(self, tickers: Optional[Iterable[str]] = None) -> None:
        """Load ticker-level JSONL files into memory."""

        if tickers is None:
            tickers = sorted(path.stem for path in self.data_dir.glob("*.jsonl") if path.stem != "all_news")

        for ticker in tickers:
            self.news[ticker] = self._load_ticker(ticker)

    def load_merged(self) -> pd.DataFrame:
        """Load the merged dataset if it exists."""

        path = self.data_dir / "all_news.jsonl"
        records = load_jsonl_records(path)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(records)
        if "published_at" in frame.columns:
            frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        return frame

    def get_raw(self, ticker: str) -> pd.DataFrame:
        """Return raw records for one ticker."""

        return self.news.get(ticker, pd.DataFrame()).copy()
