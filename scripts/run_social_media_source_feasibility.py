"""Document ethical social-media collection options for the EGX thesis.

The script intentionally avoids unofficial scraping. It can optionally collect
Reddit data only through approved OAuth credentials for Reddit's official Data
API. X is documented as not free for third-party public reads under current
official pricing, so the script records it as future work unless paid credits
or institutional access are available.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlencode

import requests


DEFAULT_SUBREDDITS = ("PersonalFinanceEgypt", "Egypt", "EgyptianStockMarket", "investing", "stocks")
DEFAULT_QUERIES = (
    "EGX",
    "EGX30",
    "البورصة المصرية",
    "البورصه المصريه",
    "البورصة",
    "CIB",
    "COMI",
    "البنك التجاري الدولي",
    "TMG",
    "TMGH",
    "طلعت مصطفى",
    "OIH",
    "اوراسكوم للاستثمار",
)


@dataclass
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run official social-media collection feasibility checks.")
    parser.add_argument("--output-dir", default="outputs/social_media_feasibility")
    parser.add_argument("--queries", nargs="*", default=list(DEFAULT_QUERIES))
    parser.add_argument("--subreddits", nargs="*", default=list(DEFAULT_SUBREDDITS))
    parser.add_argument("--limit-per-query", type=int, default=25)
    parser.add_argument(
        "--run-reddit",
        action="store_true",
        help="Use official Reddit OAuth credentials from REDDIT_CLIENT_ID/SECRET/USER_AGENT.",
    )
    return parser.parse_args()


def load_reddit_credentials() -> Optional[RedditCredentials]:
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv("REDDIT_USER_AGENT", "").strip()
    if not client_id or not client_secret or not user_agent:
        return None
    return RedditCredentials(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def reddit_token(credentials: RedditCredentials) -> str:
    auth = base64.b64encode(f"{credentials.client_id}:{credentials.client_secret}".encode("utf-8")).decode("ascii")
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": credentials.user_agent,
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def reddit_search(
    *,
    token: str,
    user_agent: str,
    subreddit: str,
    query: str,
    limit: int,
) -> List[Dict[str, object]]:
    url = f"https://oauth.reddit.com/r/{subreddit}/search"
    params = {
        "q": query,
        "restrict_sr": "true",
        "sort": "new",
        "limit": min(max(limit, 1), 100),
        "raw_json": 1,
    }
    response = requests.get(
        f"{url}?{urlencode(params)}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    rows: List[Dict[str, object]] = []
    for child in response.json().get("data", {}).get("children", []):
        data = child.get("data", {})
        rows.append(
            {
                "platform": "reddit",
                "subreddit": subreddit,
                "query": query,
                "id": data.get("id"),
                "created_utc": data.get("created_utc"),
                "title": data.get("title"),
                "selftext": data.get("selftext"),
                "score": data.get("score"),
                "num_comments": data.get("num_comments"),
                "permalink": data.get("permalink"),
                "url": data.get("url"),
            }
        )
    return rows


def run_reddit_collection(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    credentials = load_reddit_credentials()
    if credentials is None:
        return {
            "status": "SKIPPED",
            "reason": (
                "Missing REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, or REDDIT_USER_AGENT. "
                "Use only approved official Reddit API credentials."
            ),
            "records": 0,
        }

    token = reddit_token(credentials)
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    for subreddit in args.subreddits:
        for query in args.queries:
            try:
                rows.extend(
                    reddit_search(
                        token=token,
                        user_agent=credentials.user_agent,
                        subreddit=subreddit,
                        query=query,
                        limit=args.limit_per_query,
                    )
                )
                time.sleep(1.0)
            except Exception as exc:
                errors.append({"subreddit": subreddit, "query": query, "error": f"{type(exc).__name__}: {exc}"})

    raw_path = output_dir / "reddit_official_api_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "status": "COMPLETED",
        "records": len(rows),
        "errors": errors,
        "raw_path": str(raw_path),
    }


def write_report(args: argparse.Namespace, reddit_result: Dict[str, object], output_dir: Path) -> None:
    lines = [
        "# Social Media Source Feasibility",
        "",
        "This experiment evaluates whether social-media data can be added to the EGX sentiment thesis using only ethical and officially permitted access paths.",
        "",
        "## Platform Decision",
        "",
        "| Platform | Official free/ethical path | Feasibility for this thesis | Decision |",
        "| --- | --- | --- | --- |",
        "| Reddit | Possible through Reddit Data API for approved non-commercial use or Reddit's research access path. Research use must remain academic, avoid redistribution, and avoid training models on Reddit data without explicit consent. | Feasible only after approval and with strict data-handling limits. Reddit volume for EGX-specific discussion is likely sparse. | Keep as future-work or run only with approved credentials. |",
        "| X / Twitter | Official public reads currently require pay-per-use credits; owned reads are cheaper but cover only the authenticated user's own data. | Not free for broad EGX search. Unofficial scraping would be inappropriate for the thesis. | Exclude from current experiment; document as paid/institutional future work. |",
        "",
        "Official source notes used for this decision:",
        "",
        "- Reddit Help states that research use is allowed only for academic non-commercial purposes, subject to restrictions on redistribution and model training, and points researchers to Reddit's research access path.",
        "- X API documentation describes a pay-per-use model for public API access; owned reads are cheaper but limited to the authenticated user's own resources.",
        "",
        "## Reddit Collection Status",
        "",
        f"- Status: `{reddit_result.get('status')}`",
        f"- Records: `{reddit_result.get('records', 0)}`",
    ]
    if reddit_result.get("reason"):
        lines.append(f"- Reason: `{reddit_result['reason']}`")
    if reddit_result.get("raw_path"):
        lines.append(f"- Raw result path: `{reddit_result['raw_path']}`")

    lines.extend(
        [
            "",
            "## EGX Query Plan",
            "",
            "Recommended query terms combine market-level and ticker/company-level identifiers in Arabic and English.",
            "",
            "```text",
            "\n".join(args.queries),
            "```",
            "",
            "Recommended subreddits for a pilot are:",
            "",
            "```text",
            "\n".join(args.subreddits),
            "```",
            "",
            "## Modeling Design If Approved Data Is Available",
            "",
            "1. Collect posts through the official Reddit API or approved research API only.",
            "2. Store only minimal metadata needed for reproducibility; avoid redistributing raw user content in the thesis artifact.",
            "3. Apply entity matching for CIB, TMG, OIH, EGX, and Arabic aliases.",
            "4. Route Arabic/English text through the same language-aware sentiment layer used for news.",
            "5. Aggregate social sentiment daily with volume, sentiment mean, confidence, and ticker relevance features.",
            "6. Compare news-only, social-only, news+social, and ensemble models under the same chronological splits and leakage checks.",
            "",
            "## Paper-Safe Conclusion",
            "",
            "The current thesis should present social media as future work, not as a completed experiment, unless approved Reddit credentials are obtained and the official API collection is run. X should be excluded from the free implementation path because broad public-read access is paid under the current official API model.",
            "",
            "The most EGX-relevant prior work found is AlKhatib et al. (2016), which studied Arabic tweets and Arab market indices including EGX30 using Granger causality and Pearson correlation. That paper supports the relevance of Arabic social-media sentiment for future EGX-oriented work, but it does not solve the current platform-access constraint.",
        ]
    )
    (output_dir / "social_media_feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_reddit:
        reddit_result = run_reddit_collection(args, output_dir)
    else:
        reddit_result = {
            "status": "SKIPPED",
            "reason": "Reddit collection was not requested. Use --run-reddit only after obtaining approved official API credentials.",
            "records": 0,
        }
    metadata = {
        "args": vars(args),
        "reddit_result": reddit_result,
        "ethical_rule": "No unofficial scraping, no paid API bypasses, no redistribution of raw social-media data.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(args, reddit_result, output_dir)
    print("Saved social-media feasibility report to", output_dir)


if __name__ == "__main__":
    main()
