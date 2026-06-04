#!/usr/bin/env python3
"""Build the survivorship-bias-free EGX30 company registry for the five-year study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402
from egx_direction.pit_constituents import (  # noqa: E402
    DEFAULT_COMPANY_PATH,
    DEFAULT_CONSTITUENT_PATH,
    validate_constituent_registry,
)


DEFAULT_EVENTS_PATH = Path("data/egx30_rebalance_events_5yr.csv")
DEFAULT_VALIDATION_PATH = Path("data/egx30_historical_constituents_5yr.validation.json")

# The ticker column is the stable company identity used by this study. The
# Mubasher slug is the historical stock-page route used to retrieve articles.
COMPANIES: Dict[str, Tuple[str, str, str]] = {
    "ABUK": ("ABUK", "Abu Qir Fertilizers and Chemicals", "Abu Qir Fertilizers"),
    "ADIB": ("ADIB", "Abu Dhabi Islamic Bank Egypt", "ADIB Egypt"),
    "ALCN": ("ALCN", "Alexandria Container and Cargo Handling", "Alexandria Containers"),
    "AMOC": ("AMOC", "Alexandria Mineral Oils Company", "AMOC"),
    "ARCC": ("ARCC", "Arabian Cement Company", "Arabian Cement"),
    "ASPI": ("ASPI", "Aspire Capital Holding for Financial Investments", "Aspire Capital"),
    "BINV": ("BINV", "B Investments Holding", "B Investments"),
    "BTFH": ("BTFH", "Beltone Holding", "Beltone"),
    "CCAP": ("CCAP", "Qalaa Holdings", "Qalaa"),
    "CICH": ("CICH", "CI Capital Holding for Financial Investments", "CI Capital"),
    "CIEB": ("CIEB", "Credit Agricole Egypt", "Credit Agricole"),
    "CIRA": ("CIRA", "Cairo for Investment and Real Estate Development", "CIRA"),
    "CLHO": ("CLHO", "Cleopatra Hospitals Group", "Cleopatra Hospitals"),
    "COMI": ("COMI", "Commercial International Bank Egypt", "CIB|Commercial International Bank"),
    "DELTA": ("SUGR", "Delta Sugar", "Delta Sugar"),
    "EAST": ("EAST", "Eastern Company", "Eastern Co"),
    "EDBE": ("EXPA", "Export Development Bank of Egypt", "Export Development Bank"),
    "EFID": ("EFID", "Edita Food Industries", "Edita"),
    "EFIH": ("EFIH", "E-Finance for Digital and Financial Investments", "E-Finance"),
    "EGAL": ("EGAL", "Egypt Aluminum", "Egypt Aluminium"),
    "EGCH": ("EGCH", "Egyptian Chemical Industries Kima", "Kima"),
    "EGTS": ("EGTS", "Egyptian Resorts Company", "Egyptian Resorts"),
    "EKHO": ("VLMR", "Egypt Kuwait Holding", "Valmore Holding|Valmore"),
    "EMFD": ("EMFD", "Emaar Misr for Development", "Emaar Misr"),
    "ESRS": ("ESRS", "Ezz Steel", "Ezz Steel"),
    "ETEL": ("ETEL", "Telecom Egypt", "Telecom Egypt"),
    "FAIT": ("FAIT", "Faisal Islamic Bank of Egypt", "Faisal Islamic Bank"),
    "FWRY": ("FWRY", "Fawry for Banking Technology and Electronic Payment", "Fawry"),
    "GBCO": ("GBCO", "GB Corp", "GB Auto|GB Corp"),
    "GDWA": ("GDWA", "Gadwa for Industrial Development", "Gadwa"),
    "HDBK": ("HDBK", "Housing and Development Bank", "Housing Development Bank"),
    "HELI": ("HELI", "Heliopolis Company for Housing and Development", "Heliopolis Housing"),
    "HRHO": ("HRHO", "EFG Holding", "EFG Hermes|EFG Holding"),
    "ISPH": ("ISPH", "Ibnsina Pharma", "Ibnsina"),
    "JUFO": ("JUFO", "Juhayna Food Industries", "Juhayna"),
    "MASR": ("MASR", "Madinet Masr for Housing and Development", "Madinet Nasr|Madinet Masr"),
    "MCQE": ("MCQE", "Misr Cement Qena", "Misr Cement"),
    "MFPC": ("MFPC", "Misr Fertilizers Production Company Mopco", "Mopco"),
    "MTIE": ("MTIE", "MM Group for Industry and International Trade", "MM Group"),
    "OCDI": ("OCDI", "Six of October Development and Investment Sodic", "Sodic"),
    "OFH": ("OFH", "Orascom Financial Holding", "Orascom Financial"),
    "OIH": ("OIH", "Orascom Investment Holding", "Orascom Investment"),
    "ORAS": ("ORAS", "Orascom Construction", "Orascom Construction"),
    "ORHD": ("ORHD", "Orascom Development Egypt", "Orascom Development"),
    "ORWE": ("ORWE", "Oriental Weavers", "Oriental Weavers"),
    "PHAR": ("PHAR", "Egyptian International Pharmaceutical Industries", "EIPICO"),
    "PHDC": ("PHDC", "Palm Hills Developments", "Palm Hills"),
    "PIOH": ("ASPI", "Pioneers Holding Company for Financial Investment", "Pioneers Holding"),
    "PRDC": ("PRDC", "Pioneers Properties for Urban Development", "PREDCO"),
    "QNBA": ("QNBE", "QNB Alahli", "QNB Al Ahli"),
    "RAYA": ("RAYA", "Raya Holding for Financial Investments", "Raya Holding"),
    "RMDA": ("RMDA", "Tenth of Ramadan for Pharmaceutical Industries Rameda", "Rameda"),
    "SKPC": ("SKPC", "Sidi Kerir Petrochemicals", "Sidpec"),
    "SPMD": ("SPMD", "Speed Medical", "Speed Medical"),
    "SWDY": ("SWDY", "Elsewedy Electric", "El Sewedy Electric"),
    "TALM": ("TALM", "Taaleem Management Services", "Taaleem"),
    "TMGH": ("TMGH", "Talaat Moustafa Group Holding", "TMG Holding|Talaat Moustafa"),
}


SOURCES = {
    "start": "https://mondovisione.com/media-and-resources/news/egxs-periodic-market-indices-review-2021127/",
    "aug21": "https://mondovisione.com/media-and-resources/news/egxs-semi-annual-review-of-egx-indices-2021728/",
    "pioneers": "https://www.dailynewsegypt.com/2021/10/12/egx-launches-trading-on-pioneers-holding-demerged-companies/",
    "feb22": "https://mondovisione.com/media-and-resources/news/egx-index-committee-approves-the-semi-annual-periodic-review-of-the-indices-2022126/",
    "aug22": "https://www.egx.com.eg/downloads/press_release_28-07-2022_english.pdf",
    "feb23": "https://www.egypttoday.com/Article/3/122238/3-companies-enter-EGX30-after-periodic-review",
    "aug23": "https://egx.com.eg/downloads/august_2023_rebalance_en_30-07-2023_english.pdf",
    "feb24": "https://www.egx.com.eg/downloads/egx_30_rebalance_en_31-01-2024_english.pdf",
    "aug24": "https://enterpriseam.com/egypt/2024/07/31/emaar-misr-b-investments-credit-agricole-and-more-join-egx30-after-index-review/",
    "feb25": "https://mondovisione.com/media-and-resources/news/the-egyptian-exchange-egx-announces-the-semi-annual-review-of-its-market-indic-2025130/",
    "aug25": "https://mondovisione.com/media-and-resources/news/the-egyptian-exchange-egx-announces-the-semi-annual-review-of-its-market-indic-2025730/",
    "feb26": "https://mondovisione.com/_assets/files/press_release_indices_feb_2026_english_english.pdf",
}


SNAPSHOTS: Sequence[Tuple[str, str, str, str, Sequence[str]]] = (
    (
        "2021-04-21", "2021-07-31", "start", "OIH/OFH demerger transition: accepted 31-company state.",
        "CIEB COMI GBCO MTIE ORWE EAST AMOC CCAP EKHO CLHO ISPH SWDY ABUK ESRS SKPC FWRY HRHO HELI MASR ORHD PHDC TMGH ETEL PIOH OCDI EFID EMFD EDBE OIH OFH CICH".split(),
    ),
    (
        "2021-08-01", "2021-10-12", "aug21", "",
        "CIEB COMI GBCO CIRA MTIE ORWE EAST AMOC CCAP EKHO CLHO ISPH RMDA SWDY ABUK ESRS MFPC SKPC FWRY HRHO RAYA EGTS HELI MASR ORHD PHDC TMGH ETEL SPMD PIOH".split(),
    ),
    (
        "2021-10-13", "2022-01-31", "pioneers", "Pioneers demerger transition: accepted 32-company state.",
        "CIEB COMI GBCO CIRA MTIE ORWE EAST AMOC CCAP EKHO CLHO ISPH RMDA SWDY ABUK ESRS MFPC SKPC FWRY HRHO RAYA EGTS HELI MASR ORHD PHDC TMGH ETEL SPMD ASPI GDWA PRDC".split(),
    ),
    (
        "2022-02-01", "2022-07-31", "feb22", "",
        "CIEB COMI GBCO CIRA MTIE ORWE EAST AMOC CCAP EKHO CLHO ISPH RMDA SWDY ABUK ESRS MFPC SKPC FWRY HRHO HELI MASR ORHD PHDC TMGH ETEL EFIH ADIB ORAS HDBK".split(),
    ),
    (
        "2022-08-01", "2023-01-31", "aug22", "",
        "PHDC AMOC ORAS GBCO EKHO ORWE TMGH HRHO ABUK ETEL JUFO FWRY EAST EFIH COMI SKPC MFPC MASR ESRS SWDY HELI CCAP CIRA CIEB ISPH CLHO RMDA ALCN HDBK QNBA".split(),
    ),
    (
        "2023-02-01", "2023-07-31", "feb23", "",
        "PHDC AMOC ORAS GBCO EKHO ORWE TMGH HRHO ABUK ETEL JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC MASR EFID ESRS SWDY HELI CCAP CIRA CIEB TALM ISPH CLHO RMDA".split(),
    ),
    (
        "2023-08-01", "2024-01-31", "aug23", "",
        "PHDC AMOC ORAS BTFH GBCO EKHO ORWE TMGH HRHO ABUK ETEL JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC MASR ALCN EFID ESRS SWDY HELI CCAP ORHD CIRA CIEB BINV".split(),
    ),
    (
        "2024-02-01", "2024-07-31", "feb24", "",
        "PHDC AMOC ORAS BTFH GBCO EKHO ORWE TMGH HRHO ABUK ETEL JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC MASR ALCN EFID ESRS SWDY HELI CCAP ISPH ORHD DELTA PHAR".split(),
    ),
    (
        "2024-08-01", "2025-02-01", "aug24", "",
        "PHDC AMOC ORAS EMFD BTFH GBCO EKHO ORWE TMGH HRHO ABUK ETEL JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC CIEB MASR ALCN EFID ESRS BINV FAIT CLHO SWDY HELI".split(),
    ),
    (
        "2025-02-02", "2025-08-02", "feb25", "",
        "PHDC AMOC ORAS EMFD BTFH ISPH CCAP GBCO EKHO ORHD EGAL ORWE TMGH HRHO ABUK ETEL RMDA JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC CIEB MASR ALCN PHAR EFID".split(),
    ),
    (
        "2025-08-03", "2026-01-31", "aug25", "",
        "PHDC AMOC ORAS EMFD BTFH ISPH CCAP GBCO RAYA EKHO ARCC ORHD EGAL ORWE TMGH HRHO ABUK ETEL RMDA MCQE JUFO ADIB FWRY EAST EFIH COMI SKPC MFPC CIEB MASR".split(),
    ),
    (
        "2026-02-01", "2026-04-20", "feb26", "",
        "PHDC EGCH AMOC ORAS EMFD BTFH ISPH CCAP GBCO RAYA EKHO ARCC ORHD EGAL ORWE TMGH HELI HRHO ABUK ETEL OIH RMDA MCQE JUFO ADIB FWRY EAST EFID EFIH COMI".split(),
    ),
)


def build_registry() -> pd.DataFrame:
    """Materialize one row per company and point-in-time membership interval."""

    rows: List[Dict[str, str]] = []
    for date_from, date_to, source_key, note, tickers in SNAPSHOTS:
        for ticker in tickers:
            slug, name, _aliases = COMPANIES[ticker]
            rows.append(
                {
                    "Date_From": date_from,
                    "Date_To": date_to,
                    "Ticker": ticker,
                    "Mubasher_Slug": slug,
                    "Company_Name": name,
                    "Constituent_Source": SOURCES[source_key],
                    "Transition_Note": note,
                }
            )
    return pd.DataFrame(rows).sort_values(["Date_From", "Ticker"]).reset_index(drop=True)


def build_company_registry() -> pd.DataFrame:
    """Materialize the distinct historical company union for scraping."""

    rows = []
    for ticker, (slug, name, aliases) in sorted(COMPANIES.items()):
        rows.append(
            {
                "Ticker": ticker,
                "Mubasher_Slug": slug,
                "Company_Name": name,
                "Company_Name_Ar": "",
                "Aliases": "|".join(dict.fromkeys([ticker, slug, name, *aliases.split("|")])),
            }
        )
    return pd.DataFrame(rows)


def build_events() -> pd.DataFrame:
    """Materialize rebalance provenance at each encoded boundary."""

    rows = []
    previous = None
    for date_from, date_to, source_key, note, tickers in SNAPSHOTS:
        current = set(tickers)
        rows.append(
            {
                "Effective_Date": date_from,
                "Date_To": date_to,
                "Added": "|".join(sorted(current - previous)) if previous is not None else "",
                "Removed": "|".join(sorted(previous - current)) if previous is not None else "",
                "Company_Count": len(current),
                "Source": SOURCES[source_key],
                "Transition_Note": note,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate encoded rows without overwriting CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_registry()
    validation = validate_constituent_registry(registry)
    company_registry = build_company_registry()
    events = build_events()

    if not args.check_only:
        DEFAULT_CONSTITUENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        registry.to_csv(DEFAULT_CONSTITUENT_PATH, index=False)
        company_registry.to_csv(DEFAULT_COMPANY_PATH, index=False)
        events.to_csv(DEFAULT_EVENTS_PATH, index=False)
        DEFAULT_VALIDATION_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        update_pipeline_state(
            current_phase="phase_1_constituent_reconstruction",
            status="completed",
            completed_phases=["phase_1_constituent_reconstruction"],
            artifacts={
                "historical_constituents": str(DEFAULT_CONSTITUENT_PATH),
                "company_union": str(DEFAULT_COMPANY_PATH),
                "rebalance_events": str(DEFAULT_EVENTS_PATH),
                "constituent_validation": str(DEFAULT_VALIDATION_PATH),
            },
            notes=[
                "The company registry is point-in-time and survivorship-bias-free.",
                "OIH/OFH and Pioneers demerger intervals are explicit accepted transition states.",
            ],
        )

    print(json.dumps(validation, indent=2))
    print(f"company_union={len(company_registry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
