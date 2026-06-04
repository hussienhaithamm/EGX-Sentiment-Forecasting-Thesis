#!/usr/bin/env python3
"""Verify that curated thesis LaTeX files are backed by current EGX30 artifacts.

The final thesis chapters are now manually curated for structure and wording.
This script intentionally avoids overwriting them with generated prose.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402


TEX_DIR = Path("Bachelor_Thesis_Template")
METRICS_PATH = Path("data/egx30_5yr_universe_metrics.csv")
PROCESSED_PATH = Path("data/processed_news_5yr.csv")
VALIDITY_PATH = Path("data/egx30_5yr_validity_report.json")
REGISTRY_PATH = Path("data/egx30_historical_constituents_5yr.csv")
REQUIRED_TEX_FILES = (
    "abstract.tex",
    "introduction.tex",
    "background.tex",
    "literature_review.tex",
    "methodology.tex",
    "results.tex",
    "conclusion.tex",
)


def main() -> int:
    metrics = pd.read_csv(METRICS_PATH)
    articles = pd.read_csv(PROCESSED_PATH)
    registry = pd.read_csv(REGISTRY_PATH)
    validity = json.loads(VALIDITY_PATH.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_TEX_FILES if not (TEX_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing thesis files: {missing}")
    bilingual = metrics[(metrics["view"] == "bilingual") & (metrics["split"] == "test")]
    if bilingual.empty:
        raise ValueError("No bilingual test metrics found for thesis verification.")
    if validity.get("status") != "PASS":
        raise ValueError(f"Validity report is not PASS: {validity.get('status')}")
    summary = {
        "thesis_sync_mode": "curated_no_overwrite",
        "scored_articles": int(len(articles)),
        "registry_companies": int(registry["Ticker"].nunique()),
        "bilingual_metric_rows": int(len(bilingual)),
        "validity_status": validity.get("status"),
    }
    output_path = TEX_DIR / "latex_artifact_check.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_pipeline_state(
        current_phase="phase_8_latex_sync",
        status="completed",
        completed_phases=["phase_8_latex_sync"],
        artifacts={"latex_artifact_check": str(output_path), "latex_main": str(TEX_DIR / "bachelor.tex")},
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
