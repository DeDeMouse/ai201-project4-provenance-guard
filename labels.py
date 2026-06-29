"""Transparency Label Builder — turns a fused p_ai into a reader-facing label.

Implements the band table from planning.md §7. The label is driven by the
confidence *band*, not the raw float, so that (for example) 0.51 reads as
"Inconclusive" while 0.95 reads as "Very likely AI-generated" — distinctly
different categories, which is what makes the uncertainty legible to a
non-technical reader.

Band thresholds are configuration, not magic numbers buried in the mapping: each
is a named value, overridable via an environment variable so the bands can be
recalibrated against sample data without touching code. The reader-facing copy is
kept separate from the thresholds so either can be tuned independently.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Each value is the upper (exclusive) p_ai boundary of a band: a p_ai below the
# threshold falls in that band. Override via env to recalibrate, e.g.
# PG_THRESHOLD_LEANS_AI=0.80. The top band ("ai") has no upper bound.
THRESHOLDS = {
    "human": _env_float("PG_THRESHOLD_HUMAN", 0.15),
    "leans_human": _env_float("PG_THRESHOLD_LEANS_HUMAN", 0.35),
    "uncertain": _env_float("PG_THRESHOLD_UNCERTAIN", 0.65),
    "leans_ai": _env_float("PG_THRESHOLD_LEANS_AI", 0.85),
}

# Reader-facing copy per attribution category: (label, plain-language gloss).
LABELS = {
    "human": (
        "Very likely human-written",
        "Our checks strongly point to a human author.",
    ),
    "leans_human": (
        "Probably human-written",
        "Our checks lean toward a human author, with some uncertainty.",
    ),
    "uncertain": (
        "Inconclusive — we can't reliably tell",
        "Our checks disagree or are inconclusive; treat authorship as unverified.",
    ),
    "leans_ai": (
        "Probably AI-generated",
        "Our checks lean toward AI generation, with some uncertainty.",
    ),
    "ai": (
        "Very likely AI-generated",
        "Our checks strongly point to AI generation.",
    ),
}

# Bands in ascending order of p_ai. Built from THRESHOLDS so the thresholds stay
# the single source of truth; the top band is open-ended (+inf).
_ORDER = ["human", "leans_human", "uncertain", "leans_ai", "ai"]


def _bands() -> list:
    uppers = [
        THRESHOLDS["human"],
        THRESHOLDS["leans_human"],
        THRESHOLDS["uncertain"],
        THRESHOLDS["leans_ai"],
        float("inf"),
    ]
    return list(zip(_ORDER, uppers))


def label_for(p_ai: float) -> dict:
    """Map a fused p_ai in [0, 1] to an attribution + reader-facing label."""
    for attribution, upper in _bands():
        if p_ai < upper:
            label, detail = LABELS[attribution]
            return {"attribution": attribution, "label": label, "label_detail": detail}
    label, detail = LABELS["ai"]  # safety net; unreachable due to the +inf bound
    return {"attribution": "ai", "label": label, "label_detail": detail}
