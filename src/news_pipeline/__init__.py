"""Reusable news ingestion utilities for the EGX forecasting project."""

from .config import AlMobasherScraperConfig
from .features import MubasherNewsFeatureBuilder, NewsFeatureConfig
from .loader import NewsDataLoader
from .mubasher import AlMobasherScraper
from .sentiment import FinBERTConfig, FinBERTSentimentNewsFeatureBuilder
from .tickers import TARGET_COMPANIES, TargetCompany, resolve_ticker

__all__ = [
    "AlMobasherScraper",
    "AlMobasherScraperConfig",
    "FinBERTConfig",
    "FinBERTSentimentNewsFeatureBuilder",
    "MubasherNewsFeatureBuilder",
    "NewsDataLoader",
    "NewsFeatureConfig",
    "TARGET_COMPANIES",
    "TargetCompany",
    "resolve_ticker",
]
