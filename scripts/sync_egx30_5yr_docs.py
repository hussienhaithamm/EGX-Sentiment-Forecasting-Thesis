#!/usr/bin/env python3
"""Verify curated Markdown documentation against current EGX30 artifacts.

Final repository documentation is manually curated for the thesis narrative.
This script validates required artifacts and avoids overwriting polished prose.
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


OUTPUT_DIR = Path("outputs/egx30_5yr_universe_experiments")
PROCESSED_PATH = Path("data/processed_news_5yr.csv")
METRICS_PATH = Path("data/egx30_5yr_universe_metrics.csv")
VALIDITY_PATH = Path("data/egx30_5yr_validity_report.json")
REQUIRED_MARKDOWN = (
    Path("README.md"),
    Path("docs/results.md"),
    Path("docs/limitations.md"),
    Path("reports/egx30_coverage_audit.md"),
    Path("reports/egx30_baseline_comparison.md"),
    Path("reports/egx30_abstention_experiment.md"),
)


def main() -> int:
    articles = pd.read_csv(PROCESSED_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    validity = json.loads(VALIDITY_PATH.read_text(encoding="utf-8"))
    missing = [str(path) for path in REQUIRED_MARKDOWN if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing curated Markdown/report files: {missing}")
    bilingual = metrics[(metrics["view"] == "bilingual") & (metrics["split"] == "test")]
    if bilingual.empty:
        raise ValueError("No bilingual test metrics found for documentation verification.")
    if validity.get("status") != "PASS":
        raise ValueError(f"Validity report is not PASS: {validity.get('status')}")
    scoped_paths = sorted(OUTPUT_DIR.glob("*_pit_scoped_articles.csv"))
    summary = {
        "doc_sync_mode": "curated_no_overwrite",
        "scored_articles": int(len(articles)),
        "bilingual_metric_rows": int(len(bilingual)),
        "pit_scoped_files": len(scoped_paths),
        "validity_status": validity.get("status"),
    }
    output_path = Path("docs/markdown_artifact_check.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_pipeline_state(
        current_phase="phase_7_markdown_sync",
        status="completed",
        completed_phases=["phase_7_markdown_sync"],
        artifacts={"markdown_artifact_check": str(output_path)},
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
