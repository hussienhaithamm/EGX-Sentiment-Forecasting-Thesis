"""Fetch and pin EGX30 index closes for target construction only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.market_index import (  # noqa: E402
    DEFAULT_EGX30_PRICE_PATH,
    EGX30_END_DATE,
    EGX30_START_DATE,
    EGX_INDEX_ENDPOINT,
    MUBASHER_EGX30_HISTORICAL_CSV,
    fetch_egx30_prices,
    fetch_mubasher_egx30_prices,
    write_egx30_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the pinned EGX30 close series from the official EGX endpoint.")
    parser.add_argument("--output-path", default=str(DEFAULT_EGX30_PRICE_PATH))
    parser.add_argument("--start-date", default=EGX30_START_DATE)
    parser.add_argument("--end-date", default=EGX30_END_DATE)
    parser.add_argument(
        "--source",
        choices=("auto", "official", "mubasher"),
        default="auto",
        help="Use the official EGX service when reachable, or the public Mubasher historical chart CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_url = EGX_INDEX_ENDPOINT
    if args.source == "mubasher":
        frame = fetch_mubasher_egx30_prices(start_date=args.start_date, end_date=args.end_date)
        source_url = MUBASHER_EGX30_HISTORICAL_CSV
    else:
        try:
            frame = fetch_egx30_prices(start_date=args.start_date, end_date=args.end_date)
        except requests.RequestException:
            if args.source == "official":
                raise
            frame = fetch_mubasher_egx30_prices(start_date=args.start_date, end_date=args.end_date)
            source_url = MUBASHER_EGX30_HISTORICAL_CSV
    metadata = write_egx30_artifacts(
        frame,
        output_path=Path(args.output_path),
        source_url=source_url,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
