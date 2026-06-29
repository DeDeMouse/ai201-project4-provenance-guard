"""Stylometric Analyzer — Signal 1 of the Provenance Guard detection pipeline.

Pure-Python, no external calls. Given a piece of text, `analyze(text)` returns an
estimated probability that the text is AI-generated (`p_ai`) together with a
`reliability` estimate describing how much that probability can be trusted given
how much text was available to measure.

The estimate is built from four classical stylometric features:

  * burstiness                 — global variation in sentence length
  * lexical diversity          — vocabulary richness (length-robust MATTR)
  * n-gram repetition          — how often word sequences repeat
  * sentence-length regularity — local, sentence-to-sentence rhythm

Human prose tends to be *bursty* and *irregular* — a mix of long and short
sentences — while AI text is often smoother and more uniform. Each feature is
mapped to an "AI-likeness" sub-score in [0, 1] and combined into a weighted
`p_ai`. This is not a ground-truth detector; it is one statistical signal that
the aggregator later fuses with the LLM signal. The calibration centres and
slopes below are deliberately simple and meant to be tuned against sample data.
"""

from __future__ import annotations

import math
import re

VERSION = "stylometry-v1"

# --- tokenization ---------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
# Sentence/line boundaries: terminal punctuation OR newlines, so that verse —
# where the line is the natural rhythmic unit — also segments sensibly.
_BOUNDARY_RE = re.compile(r"[.!?]+|\n+")

# Editors like Word, Google Docs, and macOS/iOS silently turn a typed apostrophe
# into a curly one (U+2019), so most real prose uses it. The word regex only knows
# the straight ASCII apostrophe, so without this a curly "don't" tokenizes as
# "don" + "t" — splitting every contraction and possessive and corrupting token
# counts, lexical diversity, and repetition. Folding the common typographic
# variants to ASCII makes smart-quoted text tokenize identically to plain text.
_PUNCT_NORMALIZE = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'",  # left/right single quote, modifier apostrophe
    "“": '"', "”": '"',                  # left/right double quote
})


def _normalize(text: str) -> str:
    """Fold typographic quotes/apostrophes to ASCII so tokenization is stable."""
    return text.translate(_PUNCT_NORMALIZE)


def _words(text: str) -> list:
    """Lower-cased word tokens used for lexical and repetition features."""
    return _WORD_RE.findall(_normalize(text).lower())


def _unit_lengths(text: str) -> list:
    """Word counts per sentence/line — the series burstiness operates on."""
    counts = [len(_WORD_RE.findall(part)) for part in _BOUNDARY_RE.split(_normalize(text))]
    return [c for c in counts if c > 0]


# --- helpers --------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x <= -60.0:
        return 0.0
    if x >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --- individual features --------------------------------------------------

def _burstiness_cv(units: list):
    """Coefficient of variation (std / mean) of sentence lengths.

    High CV → varied lengths (human-like); low CV → uniform (AI-like). Returns
    None when there are too few sentences for the spread to mean anything.
    """
    n = len(units)
    if n < 2:
        return None
    mean = sum(units) / n
    if mean == 0:
        return None
    variance = sum((u - mean) ** 2 for u in units) / n
    return math.sqrt(variance) / mean


def _regularity_masd(units: list):
    """Mean absolute successive difference of sentence lengths, normalized.

    Captures *local* rhythm — do neighbouring sentences differ in length? Low
    values → a smooth, even cadence (AI-like). Distinct from burstiness, which
    measures the global spread rather than the sentence-to-sentence step.
    """
    n = len(units)
    if n < 2:
        return None
    mean = sum(units) / n
    if mean == 0:
        return None
    diffs = [abs(units[i + 1] - units[i]) for i in range(n - 1)]
    return (sum(diffs) / len(diffs)) / mean


def _lexical_diversity(words: list, window: int = 50) -> float:
    """Moving-average type-token ratio (MATTR): length-robust vocab richness.

    Higher → richer vocabulary (human-leaning); lower → more repetitive word
    choice (AI-leaning). MATTR is used instead of plain TTR because TTR falls
    purely as a function of length and would punish longer texts.
    """
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    total = 0.0
    for i in range(n - window + 1):
        total += len(set(words[i:i + window])) / window
    return total / (n - window + 1)


def _ngram_repetition(words: list, n: int = 3) -> float:
    """Fraction of n-gram occurrences that are repeats (0.0 = all unique)."""
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


# --- combination ----------------------------------------------------------

# Weights reflect how much we trust each feature as an AI signal. Burstiness is
# the strongest and best-supported; lexical diversity is the weakest (simple but
# human writing can also be low-diversity), so it carries the least weight.
_WEIGHTS = {
    "burstiness": 0.35,
    "regularity": 0.25,
    "repetition": 0.20,
    "lexical_diversity": 0.20,
}


def analyze(text: str) -> dict:
    """Estimate p(AI-generated) for `text` from stylometric features.

    Returns a dict with:
      p_ai        — probability in [0, 1] that the text is AI-generated
      reliability — trust in that estimate in [0, 1], driven by sample size
      features    — raw metric values (recorded in the audit log)
      subscores   — per-feature AI-likeness in [0, 1]
      n_tokens, n_sentences, version
    """
    words = _words(text)
    units = _unit_lengths(text)
    n_tokens = len(words)
    n_sentences = len(units)

    # Degenerate input: nothing to measure → maximal uncertainty.
    if n_tokens == 0:
        return {
            "p_ai": 0.5,
            "reliability": 0.0,
            "features": {},
            "subscores": {},
            "n_tokens": 0,
            "n_sentences": 0,
            "version": VERSION,
        }

    cv = _burstiness_cv(units)
    masd = _regularity_masd(units)
    mattr = _lexical_diversity(words)
    rep = _ngram_repetition(words)

    # Map each metric to an AI-likeness sub-score in [0, 1]. Undefined metrics
    # (e.g. a single sentence) become neutral 0.5 so they neither accuse nor
    # exonerate — the low reliability already flags the thin evidence.
    sub = {
        "burstiness": 0.5 if cv is None else _sigmoid((0.50 - cv) * 4.0),
        "regularity": 0.5 if masd is None else _sigmoid((0.50 - masd) * 4.0),
        "repetition": _sigmoid((rep - 0.08) * 12.0),
        "lexical_diversity": _sigmoid((0.70 - mattr) * 6.0),
    }

    p_ai = sum(_WEIGHTS[k] * sub[k] for k in _WEIGHTS)
    # A single heuristic should never be maximally certain on its own.
    p_ai = _clamp(p_ai, 0.05, 0.95)

    # Reliability is capped by the weakest dimension: enough sentences (for
    # burstiness/regularity) AND enough tokens (for diversity/repetition).
    r_sentences = 1.0 - math.exp(-n_sentences / 5.0)
    r_tokens = 1.0 - math.exp(-n_tokens / 120.0)
    reliability = min(r_sentences, r_tokens)

    return {
        "p_ai": round(p_ai, 4),
        "reliability": round(reliability, 4),
        "features": {
            "burstiness_cv": None if cv is None else round(cv, 4),
            "sentence_length_regularity_masd": None if masd is None else round(masd, 4),
            "lexical_diversity_mattr": round(mattr, 4),
            "ngram_repetition": round(rep, 4),
        },
        "subscores": {k: round(v, 4) for k, v in sub.items()},
        "n_tokens": n_tokens,
        "n_sentences": n_sentences,
        "version": VERSION,
    }
