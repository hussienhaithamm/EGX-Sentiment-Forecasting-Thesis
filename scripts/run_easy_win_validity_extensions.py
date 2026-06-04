#!/usr/bin/env python3
"""Generate additive easy-win validity reports for the EGX30 sentiment-only project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.easy_win_reports import (  # noqa: E402
    abstention_rows_for_panel,
    baseline_rows_for_panel,
    build_sector_daily,
    company_coverage,
    daily_coverage,
    load_frame,
    write_abstention_report,
    write_baseline_report,
    write_coverage_report,
    write_sector_report,
    write_weighting_report,
    yearly_coverage,
)
from egx_direction.market_index import DEFAULT_EGX30_PRICE_PATH, load_egx30_prices  # noqa: E402
from egx_direction.pit_constituents import (  # noqa: E402
    DEFAULT_CONSTITUENT_PATH,
    EGX30_RESEARCH_END,
    EGX30_RESEARCH_START,
    load_constituent_registry,
)


DEFAULT_OUTPUT_DIR = Path("outputs/egx30_5yr_universe_experiments")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_SECTOR_MAP = Path("data/metadata/egx30_sector_map.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--sector-map-path", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--constituent-path", type=Path, default=DEFAULT_CONSTITUENT_PATH)
    parser.add_argument("--target-price-path", type=Path, default=DEFAULT_EGX30_PRICE_PATH)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--refit-every-rows", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    constituents = load_constituent_registry(args.constituent_path)
    prices = load_egx30_prices(
        args.target_price_path,
        start_date=EGX30_RESEARCH_START,
        end_date=EGX30_RESEARCH_END,
        require_exact_bounds=True,
    )
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
    trading_dates = prices["Date"].dropna().sort_values().tolist()
    scoped_articles = _load_primary_scoped_articles(args.output_dir)
    sector_map = _ensure_sector_map(args.sector_map_path, constituents)

    company = company_coverage(scoped_articles, constituents)
    yearly = yearly_coverage(scoped_articles, trading_dates)
    daily = daily_coverage(scoped_articles, trading_dates, sector_map)
    coverage_summary = write_coverage_report(company, yearly, daily, args.report_dir)

    predictions = load_frame(args.output_dir / "predictions" / "predictions.csv")
    metrics = pd.read_csv(args.output_dir / "metrics" / "direction_metrics.csv")
    baseline_frames: List[pd.DataFrame] = []
    abstention_frames: List[pd.DataFrame] = []
    for panel_path, horizon, sentiment_source, view in _panel_inventory(args.output_dir):
        panel = load_frame(panel_path, date_columns=("Date", "target_horizon_end_date"))
        selected_c = _selected_c(metrics, horizon, sentiment_source, view)
        baseline_frames.append(
            baseline_rows_for_panel(
                panel,
                predictions,
                horizon=horizon,
                sentiment_source=sentiment_source,
                view=view,
                selected_c=selected_c,
                random_seed=args.random_seed,
                refit_every_rows=args.refit_every_rows,
            )
        )
        abstention_frames.append(
            abstention_rows_for_panel(
                panel,
                predictions,
                horizon=horizon,
                sentiment_source=sentiment_source,
                view=view,
            )
        )
    baseline_summary = write_baseline_report(pd.concat(baseline_frames, ignore_index=True), args.report_dir)
    abstention_summary = write_abstention_report(pd.concat(abstention_frames, ignore_index=True), args.report_dir)
    sector_summary = write_sector_report(build_sector_daily(scoped_articles, sector_map), args.report_dir)
    weighting_summary = write_weighting_report(scoped_articles, args.report_dir)

    summary: Dict[str, object] = {
        "coverage": coverage_summary,
        "baselines": baseline_summary,
        "abstention": abstention_summary,
        "sector": sector_summary,
        "weighting": weighting_summary,
        "report_dir": str(args.report_dir),
    }
    (args.report_dir / "egx30_easy_win_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_primary_scoped_articles(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("*_pit_scoped_articles.csv"))
    if not paths:
        raise FileNotFoundError(f"No point-in-time scoped article files found in {output_dir}")
    preferred = [path for path in paths if path.name.startswith("finbert_arabert")]
    return load_frame(preferred[0] if preferred else paths[0])


def _panel_inventory(output_dir: Path) -> List[Tuple[Path, int, str, str]]:
    panels = []
    for path in sorted((output_dir / "panels").glob("h*_panel.csv")):
        stem = path.stem.removesuffix("_panel")
        horizon_text, rest = stem.split("_", 1)
        horizon = int(horizon_text[1:])
        for view in ("english_only", "arabic_only", "bilingual"):
            suffix = f"_{view}"
            if rest.endswith(suffix):
                panels.append((path, horizon, rest[: -len(suffix)], view))
                break
    if not panels:
        raise FileNotFoundError(f"No panel files found in {output_dir / 'panels'}")
    return panels


def _selected_c(metrics: pd.DataFrame, horizon: int, sentiment_source: str, view: str) -> float:
    rows = metrics[
        (metrics["horizon"] == horizon)
        & (metrics["sentiment_source"] == sentiment_source)
        & (metrics["view"] == view)
    ]
    if rows.empty or "selected_C" not in rows:
        return 0.1
    return float(rows["selected_C"].dropna().iloc[0])


def _ensure_sector_map(path: Path, constituents: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        sector_lookup = _manual_sector_lookup()
        companies = constituents[["Ticker", "Company_Name"]].drop_duplicates().sort_values("Ticker")
        for ticker, company_name in companies.itertuples(index=False):
            sector = sector_lookup.get(str(ticker).upper(), "UNKNOWN")
            rows.append(
                {
                    "ticker": str(ticker).upper(),
                    "company_name": company_name,
                    "sector": sector,
                    "source_note": "Manual static mapping from existing company names; UNKNOWN retained when not confidently classified.",
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)
    frame = pd.read_csv(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["sector"] = frame["sector"].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")
    return frame


def _manual_sector_lookup() -> Dict[str, str]:
    return {
        "ABUK": "Materials",
        "ADIB": "Banks",
        "ALCN": "Transportation",
        "AMOC": "Energy",
        "ARCC": "Materials",
        "ASPI": "Financial Services",
        "BINV": "Financial Services",
        "BTFH": "Financial Services",
        "CCAP": "Financial Services",
        "CICH": "Financial Services",
        "CIEB": "Banks",
        "CIRA": "Education",
        "CLHO": "Healthcare",
        "COMI": "Banks",
        "DELTA": "Consumer Staples",
        "EAST": "Consumer Staples",
        "EDBE": "Banks",
        "EFID": "Consumer Staples",
        "EFIH": "Financial Technology",
        "EGAL": "Materials",
        "EGCH": "Materials",
        "EGTS": "Real Estate",
        "EKHO": "Financial Services",
        "EMFD": "Real Estate",
        "ESRS": "Materials",
        "ETEL": "Telecommunications",
        "FAIT": "Banks",
        "FWRY": "Financial Technology",
        "GBCO": "Consumer Discretionary",
        "GDWA": "Financial Services",
        "HDBK": "Banks",
        "HELI": "Real Estate",
        "HRHO": "Financial Services",
        "ISPH": "Healthcare",
        "JUFO": "Consumer Staples",
        "MASR": "Real Estate",
        "MCQE": "Materials",
        "MFPC": "Materials",
        "MTIE": "Consumer Discretionary",
        "OCDI": "Real Estate",
        "OFH": "Financial Services",
        "OIH": "Financial Services",
        "ORAS": "Industrials",
        "ORHD": "Real Estate",
        "ORWE": "Consumer Discretionary",
        "PHAR": "Healthcare",
        "PHDC": "Real Estate",
        "PIOH": "Financial Services",
        "PRDC": "Real Estate",
        "QNBA": "Banks",
        "RAYA": "Financial Services",
        "RMDA": "Healthcare",
        "SKPC": "Materials",
        "SPMD": "Healthcare",
        "SWDY": "Industrials",
        "TALM": "Education",
        "TMGH": "Real Estate",
    }


if __name__ == "__main__":
    raise SystemExit(main())
