"""Configuration helpers for the Mubasher news scraper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class AlMobasherScraperConfig:
    """Runtime configuration for the Mubasher news scraper."""

    output_dir: Path = Path("data/raw/news/al_mobasher")
    log_dir: Path = Path("logs/news")
    request_timeout_seconds: float = 20.0
    request_delay_seconds: float = 0.35
    request_delay_min_seconds: Optional[float] = None
    request_delay_max_seconds: Optional[float] = None
    max_retries: int = 3
    backoff_factor: float = 1.5
    max_pages_per_language: int = 100
    stop_after_seen_pages: int = 1
    stop_after_before_start_pages: int = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rescrape: bool = False
    include_raw_html: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AlMobasherScraperConfig":
        """Build config from environment variables."""

        return cls(
            output_dir=Path(os.getenv("AL_MOBASHER_OUTPUT_DIR", "data/raw/news/al_mobasher")),
            log_dir=Path(os.getenv("AL_MOBASHER_LOG_DIR", "logs/news")),
            request_timeout_seconds=_get_float("AL_MOBASHER_TIMEOUT_SECONDS", 20.0),
            request_delay_seconds=_get_float("AL_MOBASHER_REQUEST_DELAY_SECONDS", 0.35),
            request_delay_min_seconds=(
                _get_float("AL_MOBASHER_REQUEST_DELAY_MIN_SECONDS", 0.0)
                if os.getenv("AL_MOBASHER_REQUEST_DELAY_MIN_SECONDS") is not None
                else None
            ),
            request_delay_max_seconds=(
                _get_float("AL_MOBASHER_REQUEST_DELAY_MAX_SECONDS", 0.0)
                if os.getenv("AL_MOBASHER_REQUEST_DELAY_MAX_SECONDS") is not None
                else None
            ),
            max_retries=_get_int("AL_MOBASHER_MAX_RETRIES", 3),
            backoff_factor=_get_float("AL_MOBASHER_BACKOFF_FACTOR", 1.5),
            max_pages_per_language=_get_int("AL_MOBASHER_MAX_PAGES", 100),
            stop_after_seen_pages=_get_int("AL_MOBASHER_STOP_AFTER_SEEN_PAGES", 1),
            stop_after_before_start_pages=_get_int("AL_MOBASHER_STOP_AFTER_BEFORE_START_PAGES", 1),
            start_date=os.getenv("AL_MOBASHER_START_DATE"),
            end_date=os.getenv("AL_MOBASHER_END_DATE"),
            force_rescrape=_get_bool("AL_MOBASHER_FORCE_RESCRAPE", False),
            include_raw_html=_get_bool("AL_MOBASHER_INCLUDE_RAW_HTML", True),
            user_agent=os.getenv(
                "AL_MOBASHER_USER_AGENT",
                (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            ),
            log_level=os.getenv("AL_MOBASHER_LOG_LEVEL", "INFO").upper(),
        )
