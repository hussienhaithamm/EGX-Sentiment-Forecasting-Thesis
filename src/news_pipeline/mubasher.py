"""Incremental Mubasher news scraper for target EGX tickers."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AlMobasherScraperConfig
from .storage import load_existing_index, load_jsonl_records, write_jsonl_records
from .tickers import TARGET_COMPANIES, TargetCompany


_PAGE_COUNT_RE = re.compile(r"numPages\s*=\s*(\d+)")
_LISTING_PAGE_LINK_RE = re.compile(r"/markets/EGX/stocks/[^/]+/news/(\d+)")
_ARTICLE_ID_RE = re.compile(r"/news/(\d+)/")
_BLOCK_PATTERNS = ("captcha", "access denied", "temporarily unavailable", "verify you are human")

_LANGUAGE_SETTINGS = {
    "en": {
        "base_url": "https://english.mubasher.info",
        "label": "English",
    },
    "ar": {
        "base_url": "https://www.mubasher.info",
        "label": "Arabic",
    },
}


@dataclass
class ListingEntry:
    """Normalized information extracted from a news listing page."""

    article_id: str
    article_url: str
    title: str
    summary: Optional[str]
    section: Optional[str]
    list_page_url: str


class MubasherBlockedError(RuntimeError):
    """Raised when the scraper detects rate limiting or anti-bot blocking."""


class AlMobasherScraper:
    """Scrape Arabic and English Mubasher news pages for a fixed ticker set."""

    def __init__(
        self,
        config: Optional[AlMobasherScraperConfig] = None,
        tickers: Optional[Sequence[str]] = None,
        languages: Sequence[str] = ("en", "ar"),
    ):
        self.config = config or AlMobasherScraperConfig.from_env()
        self.tickers = [ticker.upper() for ticker in (tickers or TARGET_COMPANIES.keys())]
        self.languages = [language.lower() for language in languages]
        self.output_dir = self.config.output_dir
        self.log_dir = self.config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self.session = self._build_session()
        self._last_request_started_at = 0.0
        self.blocked_detected = False
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        self.failed_pages: List[Dict] = []
        self.failed_articles: List[Dict] = []
        self._failed_pages_path().touch(exist_ok=True)
        self._failed_articles_path().touch(exist_ok=True)

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("news_pipeline.al_mobasher")
        logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(self.log_dir / "al_mobasher.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            connect=self.config.max_retries,
            read=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=(403, 429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        return session

    def _request(self, url: str) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_started_at
        if elapsed < self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds - elapsed)

        self._last_request_started_at = time.monotonic()
        response = self.session.get(url, timeout=self.config.request_timeout_seconds)

        if self._is_blocked_response(response):
            self.blocked_detected = True
            message = f"Blocked while requesting {url} (status={response.status_code})"
            self.logger.error(message)
            raise MubasherBlockedError(message)

        response.raise_for_status()
        return response

    def _request_with_retries(
        self,
        url: str,
        *,
        resource_type: str,
        ticker: str,
        language: str,
        page_number: Optional[int] = None,
        article_id: Optional[str] = None,
    ) -> requests.Response:
        attempts = max(int(self.config.max_retries), 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._request(url)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break

                sleep_seconds = self._retry_sleep_seconds(attempt, blocked=isinstance(exc, MubasherBlockedError))
                self.logger.warning(
                    "Retrying %s for %s %s page=%s article=%s after attempt %s/%s failed: %s. Sleeping %.1fs",
                    resource_type,
                    ticker,
                    language,
                    page_number,
                    article_id,
                    attempt,
                    attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        assert last_exc is not None
        raise last_exc

    def _retry_sleep_seconds(self, attempt: int, *, blocked: bool = False) -> float:
        base_delay = max(float(self.config.backoff_factor), 0.1)
        delay = base_delay * (2 ** max(attempt - 1, 0))
        if blocked:
            delay *= 2.0
        return min(delay, 90.0)

    def _append_jsonl_record(self, path: Path, record: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _failed_pages_path(self) -> Path:
        return self.output_dir / "failed_pages.jsonl"

    def _failed_articles_path(self) -> Path:
        return self.output_dir / "failed_articles.jsonl"

    @staticmethod
    def _exception_payload(exc: BaseException) -> Dict[str, str]:
        return {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    def _record_failed_page(
        self,
        company: TargetCompany,
        language: str,
        page_number: int,
        page_url: str,
        exc: BaseException,
        *,
        attempts: Optional[int] = None,
        page_count: Optional[int] = None,
    ) -> None:
        payload = {
            "run_started_at": self.run_started_at,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "resource_type": "listing_page",
            "ticker": company.ticker,
            "canonical_ticker": company.ticker,
            "site_stock_code": company.site_code,
            "company_name_en": company.company_name_en,
            "company_name_ar": company.company_name_ar,
            "language": language,
            "page_number": page_number,
            "page_count": page_count,
            "url": page_url,
            "attempts": attempts or max(int(self.config.max_retries), 1),
            "blocked_detected": isinstance(exc, MubasherBlockedError),
            **self._exception_payload(exc),
        }
        self.failed_pages.append(payload)
        self._append_jsonl_record(self._failed_pages_path(), payload)
        self.logger.error(
            "Skipping failed listing page after retries: %s %s page %s/%s %s (%s: %s)",
            company.ticker,
            language,
            page_number,
            page_count or "?",
            page_url,
            payload["error_type"],
            payload["error_message"],
        )

    def _record_failed_article(
        self,
        company: TargetCompany,
        language: str,
        listing_entry: ListingEntry,
        exc: BaseException,
    ) -> None:
        payload = {
            "run_started_at": self.run_started_at,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "resource_type": "article_page",
            "ticker": company.ticker,
            "canonical_ticker": company.ticker,
            "site_stock_code": company.site_code,
            "company_name_en": company.company_name_en,
            "company_name_ar": company.company_name_ar,
            "language": language,
            "article_id": listing_entry.article_id,
            "url": listing_entry.article_url,
            "listing_title": listing_entry.title,
            "list_page_url": listing_entry.list_page_url,
            "attempts": max(int(self.config.max_retries), 1),
            "blocked_detected": isinstance(exc, MubasherBlockedError),
            **self._exception_payload(exc),
        }
        self.failed_articles.append(payload)
        self._append_jsonl_record(self._failed_articles_path(), payload)
        self.logger.error(
            "Skipping failed article after retries: %s %s article %s %s (%s: %s)",
            company.ticker,
            language,
            listing_entry.article_id,
            listing_entry.article_url,
            payload["error_type"],
            payload["error_message"],
        )

    @staticmethod
    def _is_blocked_response(response: requests.Response) -> bool:
        if response.status_code in {403, 429, 503}:
            return True
        lowered = response.text.lower()
        return any(pattern in lowered for pattern in _BLOCK_PATTERNS)

    @staticmethod
    def _extract_article_id(url: str) -> str:
        match = _ARTICLE_ID_RE.search(url)
        return match.group(1) if match else ""

    @staticmethod
    def _clean_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned or None

    @staticmethod
    def _parse_listing_blocks(soup: BeautifulSoup, page_url: str, base_url: str) -> List[ListingEntry]:
        entries: List[ListingEntry] = []
        for block in soup.select(".mi-article-media-block"):
            title_node = block.select_one(".mi-article-media-block__title")
            if not title_node or not title_node.get("href"):
                continue
            article_url = urljoin(base_url, title_node["href"])
            article_id = AlMobasherScraper._extract_article_id(article_url)
            if not article_id:
                continue
            summary_node = block.select_one(".mi-article-media-block__text")
            section_node = block.select_one(".global__section-label")
            entries.append(
                ListingEntry(
                    article_id=article_id,
                    article_url=article_url,
                    title=AlMobasherScraper._clean_text(title_node.get_text(" ", strip=True)) or "",
                    summary=AlMobasherScraper._clean_text(
                        summary_node.get_text(" ", strip=True) if summary_node else None
                    ),
                    section=AlMobasherScraper._clean_text(
                        section_node.get_text(" ", strip=True) if section_node else None
                    ),
                    list_page_url=page_url,
                )
            )
        return entries

    @staticmethod
    def _extract_page_count(html: str) -> Optional[int]:
        match = _PAGE_COUNT_RE.search(html)
        if match:
            return int(match.group(1))
        page_numbers = [int(value) for value in _LISTING_PAGE_LINK_RE.findall(html) if value.isdigit()]
        return max(page_numbers) if page_numbers else None

    @staticmethod
    def _select_first_text(nodes: Iterable[Tag]) -> Optional[str]:
        for node in nodes:
            if not isinstance(node, Tag):
                continue
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if text and "{{" not in text and "}}" not in text:
                return text
        return None

    @staticmethod
    def _extract_article_body(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], bool]:
        container = soup.select_one(".article__content-text")
        if container is None:
            container = soup.select_one('.mi-article__content[itemprop="articleBody"]')
        if container is None:
            return None, None, False

        container = BeautifulSoup(str(container), "html.parser")
        root = container.find()
        if root is None:
            return None, None, False

        for selector in (
            ".mi-article__stocks",
            ".stock-price-block",
            ".outstream-ad-container",
            ".display_center_align",
            ".mi-article__tags",
            ".mi-article__attachments",
            ".mi-article__related-news",
            ".mi-hide-for-small",
            "script",
            "style",
            "iframe",
        ):
            for node in root.select(selector):
                node.decompose()

        paragraphs = [
            re.sub(r"\s+", " ", paragraph.get_text(" ", strip=True)).strip()
            for paragraph in root.select("p")
        ]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        if paragraphs:
            body_text = "\n\n".join(paragraphs)
        else:
            body_text = re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()

        lowered_body = body_text.lower()
        paywalled = "exclusive services" in lowered_body or "الخدمة الاخبار المدفوعة" in body_text
        raw_html = str(root)
        return body_text or None, raw_html or None, paywalled

    @staticmethod
    def _extract_meta_content(soup: BeautifulSoup, key: str, attr_name: str = "name") -> Optional[str]:
        node = soup.find("meta", attrs={attr_name: key})
        if not node:
            return None
        return AlMobasherScraper._clean_text(node.get("content"))

    @staticmethod
    def _extract_published_at(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
        time_node = soup.select_one('time[itemprop="datePublished"]')
        if not time_node:
            return None, None

        raw_text = AlMobasherScraper._clean_text(time_node.get_text(" ", strip=True))
        raw_datetime = time_node.get("datetime")
        if raw_datetime:
            try:
                parsed = datetime.strptime(raw_datetime, "%a %b %d %H:%M:%S UTC %Y").replace(
                    tzinfo=timezone.utc
                )
                return parsed.isoformat(), raw_text
            except ValueError:
                pass
        return None, raw_text

    def _parse_article_page(
        self,
        company: TargetCompany,
        language: str,
        listing_entry: ListingEntry,
    ) -> Dict:
        response = self._request_with_retries(
            listing_entry.article_url,
            resource_type="article_page",
            ticker=company.ticker,
            language=language,
            article_id=listing_entry.article_id,
        )
        soup = BeautifulSoup(response.text, "html.parser")

        title = self._select_first_text(soup.select("h1")) or listing_entry.title
        body, raw_body_html, is_paywalled = self._extract_article_body(soup)
        meta_description = self._extract_meta_content(soup, "description")
        published_at, published_at_raw = self._extract_published_at(soup)
        canonical_url = None
        canonical_node = soup.find("link", attrs={"rel": "canonical"})
        if canonical_node:
            canonical_url = canonical_node.get("href")

        title_hits = company.match_text(title)
        body_hits = company.match_text(body or "")
        match_methods = ["stock_page"]
        if title_hits["primary"] or title_hits["secondary"]:
            match_methods.append("title_alias")
        if body_hits["primary"] or body_hits["secondary"]:
            match_methods.append("body_alias")

        if title_hits["primary"]:
            match_confidence = "high"
        elif body_hits["primary"] or title_hits["secondary"] or body_hits["secondary"]:
            match_confidence = "medium"
        else:
            match_confidence = "low"

        summary_candidates = [listing_entry.summary, meta_description]
        summary = max(
            (candidate for candidate in summary_candidates if candidate),
            key=len,
            default=None,
        )

        article_url = canonical_url or response.url or listing_entry.article_url
        record = {
            "article_id": listing_entry.article_id,
            "ticker": company.ticker,
            "site_stock_code": company.site_code,
            "company_name": company.company_name_en if language == "en" else company.company_name_ar,
            "company_name_en": company.company_name_en,
            "company_name_ar": company.company_name_ar,
            "article_title": title,
            "article_body": body,
            "article_summary": summary,
            "publication_timestamp": published_at,
            "published_at": published_at,
            "published_at_raw": published_at_raw,
            "article_url": article_url,
            "article_language": language,
            "article_language_name": _LANGUAGE_SETTINGS[language]["label"],
            "source": "Al Mobasher",
            "section": listing_entry.section,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "is_paywalled": is_paywalled,
            "match_confidence": match_confidence,
            "match_methods": match_methods,
            "matched_primary_aliases": sorted(
                set(title_hits["primary"]) | set(body_hits["primary"])
            ),
            "matched_secondary_aliases": sorted(
                set(title_hits["secondary"]) | set(body_hits["secondary"])
            ),
            "raw": {
                "listing_title": listing_entry.title,
                "listing_summary": listing_entry.summary,
                "page_title": self._clean_text(soup.title.get_text(" ", strip=True)) if soup.title else None,
                "meta_description": meta_description,
                "list_page_url": listing_entry.list_page_url,
                "canonical_url": canonical_url,
            },
        }

        if self.config.include_raw_html:
            record["raw"]["body_html"] = raw_body_html
        return record

    def _ticker_output_path(self, ticker: str) -> Path:
        return self.output_dir / f"{ticker}.jsonl"

    def _load_existing_records(self, ticker: str) -> Dict[Tuple[str, str], Dict]:
        return load_existing_index(self._ticker_output_path(ticker))

    def _get_listing_url(self, company: TargetCompany, language: str, page_number: int) -> str:
        base_url = _LANGUAGE_SETTINGS[language]["base_url"]
        return f"{base_url}/markets/EGX/stocks/{company.site_code}/news/{page_number}"

    def _fetch_listing_page(
        self,
        company: TargetCompany,
        language: str,
        page_number: int,
    ) -> Tuple[BeautifulSoup, str]:
        page_url = self._get_listing_url(company, language, page_number)
        response = self._request_with_retries(
            page_url,
            resource_type="listing_page",
            ticker=company.ticker,
            language=language,
            page_number=page_number,
        )
        return BeautifulSoup(response.text, "html.parser"), response.text

    def _persist_ticker_progress(self, ticker: str, records: Iterable[Dict]) -> None:
        write_jsonl_records(self._ticker_output_path(ticker), records)
        self._persist_combined_news()
        self._write_manifest()

    def _persist_combined_news(self) -> None:
        merged_records: List[Dict] = []
        for ticker in sorted(self.tickers):
            merged_records.extend(load_jsonl_records(self._ticker_output_path(ticker)))
        write_jsonl_records(self.output_dir / "all_news.jsonl", merged_records)

    def _write_manifest(self) -> Dict:
        manifest = self._build_manifest()
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def scrape_company(self, company: TargetCompany) -> List[Dict]:
        existing_records = self._load_existing_records(company.ticker)
        merged_records = dict(existing_records)
        stats = {"new_records": 0, "seen_records": 0, "failed_pages": 0, "failed_articles": 0}

        for language in self.languages:
            if language not in _LANGUAGE_SETTINGS:
                raise ValueError(f"Unsupported language: {language}")

            first_page_url = self._get_listing_url(company, language, 1)
            try:
                first_soup, first_html = self._fetch_listing_page(company, language, 1)
            except Exception as exc:
                self._record_failed_page(company, language, 1, first_page_url, exc)
                stats["failed_pages"] += 1
                continue

            available_page_count = self._extract_page_count(first_html)
            page_count_source = "site" if available_page_count else "config_limit"
            available_page_count = available_page_count or self.config.max_pages_per_language
            page_count = min(available_page_count, self.config.max_pages_per_language)
            should_stop_after_seen_pages = (
                self.config.stop_after_seen_pages > 0
                and not self.config.force_rescrape
                and self.config.max_pages_per_language < available_page_count
            )
            self.logger.info(
                "Scraping %s %s (%s pages; available=%s source=%s)",
                company.ticker,
                language,
                page_count,
                available_page_count,
                page_count_source,
            )

            seen_pages_in_a_row = 0
            for page_number in range(1, page_count + 1):
                if page_number == 1:
                    page_soup = first_soup
                    page_url = first_page_url
                else:
                    page_url = self._get_listing_url(company, language, page_number)
                    try:
                        page_soup, _page_html = self._fetch_listing_page(company, language, page_number)
                    except Exception as exc:
                        self._record_failed_page(
                            company,
                            language,
                            page_number,
                            page_url,
                            exc,
                            page_count=page_count,
                        )
                        stats["failed_pages"] += 1
                        self._persist_ticker_progress(company.ticker, merged_records.values())
                        continue

                entries = self._parse_listing_blocks(
                    page_soup,
                    page_url=page_url,
                    base_url=_LANGUAGE_SETTINGS[language]["base_url"],
                )
                if not entries:
                    empty_exc = RuntimeError("No listing entries found")
                    if page_count_source == "config_limit":
                        self.logger.info("No listing entries found for %s on page %s; ending fallback page scan", company.ticker, page_url)
                        break
                    self._record_failed_page(
                        company,
                        language,
                        page_number,
                        page_url,
                        empty_exc,
                        attempts=1,
                        page_count=page_count,
                    )
                    stats["failed_pages"] += 1
                    self._persist_ticker_progress(company.ticker, merged_records.values())
                    continue

                new_records_on_page = 0
                for entry in entries:
                    key = (entry.article_id, language)
                    if key in merged_records and not self.config.force_rescrape:
                        stats["seen_records"] += 1
                        continue

                    try:
                        record = self._parse_article_page(company, language, entry)
                    except Exception as exc:
                        self._record_failed_article(company, language, entry, exc)
                        stats["failed_articles"] += 1
                        continue

                    merged_records[key] = record
                    new_records_on_page += 1
                    stats["new_records"] += 1

                self._persist_ticker_progress(company.ticker, merged_records.values())

                if should_stop_after_seen_pages and new_records_on_page == 0:
                    seen_pages_in_a_row += 1
                    if seen_pages_in_a_row >= self.config.stop_after_seen_pages:
                        self.logger.info(
                            "Stopping %s %s after %s fully-seen page(s)",
                            company.ticker,
                            language,
                            seen_pages_in_a_row,
                        )
                        break
                else:
                    seen_pages_in_a_row = 0

        records = list(merged_records.values())
        self._persist_ticker_progress(company.ticker, records)
        self.logger.info(
            "Finished %s with %s total records (%s new, %s already present, %s failed pages, %s failed articles)",
            company.ticker,
            len(records),
            stats["new_records"],
            stats["seen_records"],
            stats["failed_pages"],
            stats["failed_articles"],
        )
        return records

    def run(self) -> Dict[str, List[Dict]]:
        """Scrape all requested target tickers and write outputs."""

        results: Dict[str, List[Dict]] = {}
        for ticker in self.tickers:
            company = TARGET_COMPANIES.get(ticker)
            if company is None:
                raise KeyError(f"Unsupported target ticker: {ticker}")
            results[ticker] = self.scrape_company(company)

        self._persist_combined_news()
        self._write_manifest()

        return results

    def _build_manifest(self) -> Dict:
        summary = {}
        for ticker in self.tickers:
            records = load_jsonl_records(self._ticker_output_path(ticker))
            language_counts: Dict[str, int] = {}
            for record in records:
                language = record.get("article_language", "")
                language_counts[language] = language_counts.get(language, 0) + 1
            summary[ticker] = {
                "total_records": len(records),
                "language_counts": language_counts,
            }

        return {
            "source": "Al Mobasher",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "blocked_detected": self.blocked_detected,
            "tickers": self.tickers,
            "languages": self.languages,
            "failed_pages_path": str(self._failed_pages_path()),
            "failed_articles_path": str(self._failed_articles_path()),
            "failed_pages_current_run": len(self.failed_pages),
            "failed_articles_current_run": len(self.failed_articles),
            "failed_pages_total": len(load_jsonl_records(self._failed_pages_path())),
            "failed_articles_total": len(load_jsonl_records(self._failed_articles_path())),
            "config": {
                "output_dir": str(self.output_dir),
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "request_delay_seconds": self.config.request_delay_seconds,
                "max_retries": self.config.max_retries,
                "backoff_factor": self.config.backoff_factor,
                "max_pages_per_language": self.config.max_pages_per_language,
                "stop_after_seen_pages": self.config.stop_after_seen_pages,
                "force_rescrape": self.config.force_rescrape,
                "include_raw_html": self.config.include_raw_html,
            },
            "summary": summary,
        }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Scrape Al Mobasher news for target EGX tickers.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=sorted(TARGET_COMPANIES.keys()),
        help="Target project tickers to scrape.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "ar"],
        choices=sorted(_LANGUAGE_SETTINGS.keys()),
        help="Article languages to scrape.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing JSONL files and re-fetch every page up to the configured limit.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit override per ticker/language.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = AlMobasherScraperConfig.from_env()
    if args.force:
        config = dataclass_replace(config, force_rescrape=True)
    if args.max_pages is not None:
        config = dataclass_replace(config, max_pages_per_language=args.max_pages)
    if args.output_dir is not None:
        config = dataclass_replace(config, output_dir=Path(args.output_dir))

    scraper = AlMobasherScraper(config=config, tickers=args.tickers, languages=args.languages)
    scraper.run()
    return 0


def dataclass_replace(config: AlMobasherScraperConfig, **kwargs) -> AlMobasherScraperConfig:
    """Small local replacement helper to avoid importing `dataclasses.replace` repeatedly."""

    values = asdict(config)
    values.update(kwargs)
    if "output_dir" in values and not isinstance(values["output_dir"], Path):
        values["output_dir"] = Path(values["output_dir"])
    if "log_dir" in values and not isinstance(values["log_dir"], Path):
        values["log_dir"] = Path(values["log_dir"])
    return AlMobasherScraperConfig(**values)
