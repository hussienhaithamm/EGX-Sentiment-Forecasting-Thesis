"""JSONL storage helpers for scraped news datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def record_sort_key(record: Dict) -> Tuple[str, str, str, str]:
    """Stable ordering for output files."""

    return (
        record.get("published_at", ""),
        record.get("ticker", ""),
        record.get("article_language", ""),
        record.get("article_id", ""),
    )


def load_jsonl_records(path: Path) -> List[Dict]:
    """Load records from a JSONL file if it exists."""

    if not path.exists():
        return []

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_existing_index(path: Path) -> Dict[Tuple[str, str], Dict]:
    """Index existing records by article id and language."""

    records = load_jsonl_records(path)
    index: Dict[Tuple[str, str], Dict] = {}
    for record in records:
        key = (str(record.get("article_id", "")), record.get("article_language", ""))
        if key == ("", ""):
            continue
        index[key] = record
    return index


def write_jsonl_records(path: Path, records: Iterable[Dict]) -> None:
    """Write JSONL records with UTF-8 preservation and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=record_sort_key, reverse=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in ordered_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)
