"""Detection Orchestrator — runs the signals, fuses them, and labels the result.

This is the spine of planning.md's multi-signal pipeline: it fans `text` out to
the stylometric signal and the LLM signal, hands their outputs to the confidence
aggregator, and maps the fused score to a transparency label. The returned
decision dict is what the endpoint returns to the caller and records in the audit
log.
"""

from __future__ import annotations

import aggregator
import labels
import llm_detector
import stylometry


def classify(text: str) -> dict:
    """Run the full detection pipeline on `text` and return a decision dict."""
    stylometric = stylometry.analyze(text)
    llm = llm_detector.analyze(text)

    signals = [
        {"name": "stylometry", **stylometric},
        {"name": "llm", **llm},
    ]

    fused = aggregator.aggregate(signals)
    label = labels.label_for(fused["p_ai"])

    return {
        "signals": signals,
        "confidence": fused["p_ai"],
        "p_ai": fused["p_ai"],
        "disagreement": fused["disagreement"],
        "used_signals": fused["used_signals"],
        "attribution": label["attribution"],
        "label": label["label"],
        "label_detail": label["label_detail"],
        "model_version": f"{stylometry.VERSION}+{llm.get('model', 'n/a')}",
    }
