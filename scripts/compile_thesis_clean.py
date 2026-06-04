#!/usr/bin/env python3
"""Compile the thesis through BibTeX and fail on unresolved or box warnings."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egx_direction.pipeline_state import update_pipeline_state  # noqa: E402


TEX_DIR = Path("Bachelor_Thesis_Template")
MAIN = "bachelor"
FAIL_PATTERNS = (
    r"LaTeX Warning:",
    r"Package .* Warning:",
    r"Overfull \\hbox",
    r"Underfull \\hbox",
    r"undefined references",
    r"Citation .* undefined",
    r"Reference .* undefined",
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=TEX_DIR, check=True)


def main() -> int:
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{MAIN}.tex"])
    run(["bibtex", MAIN])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{MAIN}.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{MAIN}.tex"])
    log = (TEX_DIR / f"{MAIN}.log").read_text(encoding="utf-8", errors="replace")
    matches = [pattern for pattern in FAIL_PATTERNS if re.search(pattern, log, flags=re.IGNORECASE)]
    if matches:
        raise RuntimeError(f"Thesis log contains forbidden warning patterns: {matches}")
    pdf = TEX_DIR / f"{MAIN}.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise RuntimeError("Compiled thesis PDF is missing or empty.")
    update_pipeline_state(
        current_phase="phase_9_clean_pdf_build",
        status="completed",
        completed_phases=["phase_9_clean_pdf_build"],
        artifacts={"compiled_thesis_pdf": str(pdf)},
    )
    print(f"PASS: {pdf} ({pdf.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
