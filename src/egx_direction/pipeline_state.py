"""Small helpers for resumable EGX30 research pipeline state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


DEFAULT_STATE_PATH = Path("data/pipeline_state.json")


def load_pipeline_state(path: Path = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    """Load the persisted pipeline state if present."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_pipeline_state(
    *,
    current_phase: str,
    status: str,
    path: Path = DEFAULT_STATE_PATH,
    completed_phases: Optional[Iterable[str]] = None,
    artifacts: Optional[Mapping[str, object]] = None,
    notes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Persist a phase checkpoint without discarding prior artifacts or notes."""

    state = load_pipeline_state(path)
    state.setdefault("pipeline", "egx30_5yr_point_in_time")
    state.setdefault("timeline", {"start_date": "2021-04-21", "end_date": "2026-04-20"})
    state.setdefault("completed_phases", [])
    state.setdefault("artifacts", {})
    state.setdefault("notes", [])
    state["current_phase"] = current_phase
    state["status"] = status
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    if completed_phases is not None:
        state["completed_phases"] = list(dict.fromkeys([*state["completed_phases"], *completed_phases]))
    if artifacts is not None:
        state["artifacts"].update({str(key): value for key, value in artifacts.items()})
    if notes is not None:
        state["notes"] = list(dict.fromkeys([*state["notes"], *notes]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state
