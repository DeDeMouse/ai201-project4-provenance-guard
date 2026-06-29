"""Structured audit log for Provenance Guard.

Every attribution decision (and, later, every appeal) is appended as one JSON
object per line to an append-only log file (JSON Lines). JSONL keeps the log
structured and machine-readable while remaining trivial and robust to append to:
a single line write, where a crash at worst corrupts only the final line.

This is deliberately the simplest thing that satisfies the requirement — a
structured, replayable record rather than `print()` output. The backing store
can later be swapped for SQLite without changing the `record()` / `get_log()`
contract the rest of the app depends on.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

# Log location is configurable so tests and deployments can redirect it; it
# defaults to a file alongside this module.
LOG_PATH = os.environ.get(
    "AUDIT_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.jsonl"),
)

# Serialise writes/reads so concurrent requests can't interleave a line.
_lock = threading.Lock()


def record(entry: dict) -> dict:
    """Append a structured entry to the audit log and return the stored record.

    A UTC ISO-8601 `timestamp` is added if the caller did not supply one.
    """
    stored = dict(entry)
    stored.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    line = json.dumps(stored, ensure_ascii=False)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return stored


def get_log(limit: int = 50) -> list:
    """Return up to `limit` of the most recent entries, newest first.

    Malformed lines (e.g. from a crash mid-write) are skipped rather than
    failing the whole read.
    """
    if not os.path.exists(LOG_PATH):
        return []
    entries = []
    with _lock:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    entries.reverse()
    return entries[: max(0, limit)]
