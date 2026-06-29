"""Confidence Aggregator — fuses detection signals into one calibrated score.

Implements planning.md §6 (Uncertainty representation): each signal reports a
probability `p_ai` that the text is AI-generated plus a `reliability`. We take a
weighted mean of the usable signals and, when they disagree, pull the result
toward 0.5 so genuine conflict reads as uncertainty rather than letting one signal
win outright.

Two calibration details make that fusion honest (see also planning.md §9):

  * Per-signal trust priors. A signal's influence is its reported `reliability`
    scaled by a trust prior. Stylometry is a weaker *discriminator* than the LLM:
    its reliability reflects how much text it measured, not whether those features
    actually separate human from AI. Well-edited AI with varied sentence lengths
    reads as "human" to stylometry, so we trust it less — otherwise a confidently
    wrong statistical signal can outvote a reliable semantic one.

  * A gentle disagreement shrink. Disagreement still widens uncertainty, but via
    sqrt(1 - disagreement) rather than a linear (1 - disagreement) pull. The
    linear version over-penalised exactly the useful case — a reliable signal
    standing out against a wrong weak one — and could even rank a clearly-AI text
    below a genuinely ambiguous one. The softer curve preserves the §6 intent
    without collapsing a confident, reliable verdict to 0.5.
"""

from __future__ import annotations

import math

# Trust priors per signal (calibration, not magic numbers — tune against samples).
_TRUST = {
    "stylometry": 0.5,
    "llm": 1.0,
}
_DEFAULT_TRUST = 1.0

# Lower exponent => gentler shrink toward 0.5 on disagreement. 1.0 reproduces the
# old linear pull; 0.5 (sqrt) is the calibrated default.
_SHRINK_EXPONENT = 0.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _weight(signal: dict) -> float:
    """Fusion weight = reported reliability scaled by the signal's trust prior."""
    reliability = signal.get("reliability", 0.0)
    return reliability * _TRUST.get(signal.get("name"), _DEFAULT_TRUST)


def aggregate(signals: list) -> dict:
    """Fuse signal dicts (each with `name`, `p_ai`, `reliability`) into one score.

    Returns {"p_ai", "disagreement", "used_signals"}, where `p_ai` is the fused
    probability that the text is AI-generated. Signals with zero effective weight
    (e.g. an unavailable LLM call, reliability 0) are ignored.
    """
    usable = [s for s in signals if _weight(s) > 0]

    if not usable:
        # No trustworthy evidence at all → maximal uncertainty.
        return {"p_ai": 0.5, "disagreement": 0.0, "used_signals": 0}

    total_w = sum(_weight(s) for s in usable)
    mean = sum(s["p_ai"] * _weight(s) for s in usable) / total_w

    if len(usable) == 1:
        return {
            "p_ai": round(_clamp(mean, 0.02, 0.98), 4),
            "disagreement": 0.0,
            "used_signals": 1,
        }

    # Weight-weighted spread of the signals around the mean. The maximum possible
    # std is 0.5 (values split between 0 and 1), so 2*std normalises disagreement
    # to [0, 1].
    variance = sum(_weight(s) * (s["p_ai"] - mean) ** 2 for s in usable) / total_w
    disagreement = min(1.0, 2.0 * math.sqrt(variance))

    # Shrink the deviation from 0.5 in proportion to disagreement, but gently:
    # full agreement preserves the mean; high disagreement softens it toward 0.5
    # without flattening a confident, reliable signal outright.
    fused = 0.5 + (mean - 0.5) * ((1.0 - disagreement) ** _SHRINK_EXPONENT)

    return {
        "p_ai": round(_clamp(fused, 0.02, 0.98), 4),
        "disagreement": round(disagreement, 4),
        "used_signals": len(usable),
    }
