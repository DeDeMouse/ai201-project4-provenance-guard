"""Content Repository — persists submissions and their attribution status.

planning.md component #4 / §5 data model. A minimal JSON-file store mapping
`content_id` -> content record. Each record holds a snapshot of the original
classification plus a *mutable* `status` ("classified" -> "under_review"), which
is what the appeals workflow needs somewhere to update — the audit log is
append-only and not the place to mutate state.

The whole file is rewritten on each change under a lock. That is plenty at this
scale; the contract (`save_content` / `get_content` / `add_appeal`) is what the
rest of the app depends on and can move to SQLite later without callers changing.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

STORE_PATH = os.environ.get(
    "CONTENT_STORE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_store.json"),
)

STATUS_CLASSIFIED = "classified"
STATUS_UNDER_REVIEW = "under_review"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    # Write-then-rename so a crash mid-write can't truncate the live file.
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, STORE_PATH)


def save_content(content_id: str, creator_id: str, decision: dict, text_length: int) -> dict:
    """Persist a new submission with a snapshot of its original classification."""
    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "status": STATUS_CLASSIFIED,
        "submitted_at": _now(),
        "text_length": text_length,
        "decision": {
            "confidence": decision["confidence"],
            "attribution": decision["attribution"],
            "label": decision["label"],
            "disagreement": decision["disagreement"],
            "model_version": decision["model_version"],
        },
        "appeals": [],
    }
    with _lock:
        data = _load()
        data[content_id] = record
        _save(data)
    return record


def get_content(content_id: str):
    """Return the content record for `content_id`, or None if unknown."""
    with _lock:
        return _load().get(content_id)


def add_appeal(content_id: str, appeal: dict):
    """Attach an appeal and move the content to 'under_review'.

    Returns the updated record, or None if `content_id` is unknown.
    """
    with _lock:
        data = _load()
        record = data.get(content_id)
        if record is None:
            return None
        record["status"] = STATUS_UNDER_REVIEW
        record.setdefault("appeals", []).append(appeal)
        data[content_id] = record
        _save(data)
        return record
