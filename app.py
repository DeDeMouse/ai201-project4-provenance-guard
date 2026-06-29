import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
import pipeline
import store

app = Flask(__name__)
# Preserve the field order we set in each JSON response instead of sorting keys
# alphabetically (Flask's default), so payloads read top-to-bottom as intended.
app.json.sort_keys = False

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Upper bound on accepted submissions (characters). Oversized payloads are
# rejected to protect the pipeline. There is deliberately no lower bound beyond
# "non-empty": short creative writing is valid, and the analyzer's reliability
# score already reflects that a short sample yields a weak signal.
MAX_TEXT_CHARS = 50_000


@app.route("/")
def home():
    return "Provenance Guard is running."


@app.route("/submit", methods=["POST"])
# Layered limits sized for a real writer (per IP), not a script. Because each
# submission triggers a paid LLM call, these also cap cost:
#   10/minute — burst headroom for revising and resubmitting, plus the odd retry
#               after a 400/429, while a continuous script is cut off in seconds;
#   60/hour   — holds a sustained flood to ~1/min, already very active for writing,
#               so no one can ride the per-minute limit indefinitely;
#   200/day   — overall ceiling, well above any genuine single-author daily volume.
@limiter.limit("10 per minute; 60 per hour; 200 per day")
def submit():
    data = request.get_json(silent=True)
    if data is None:
        # Body was missing, had the wrong Content-Type, or — most commonly when
        # testing by hand — was malformed because a quote/apostrophe broke the
        # client's JSON (e.g. an apostrophe inside a single-quoted `curl -d '...'`).
        return jsonify({
            "error": "Could not parse request body as JSON. Send Content-Type: "
                     "application/json with a well-formed object. If you are using "
                     "curl, avoid single-quoting JSON that contains apostrophes — "
                     "use --data @file.json or the included submit.py client.",
        }), 400
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = data.get("text")
    creator_id = data.get("creator_id")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "Field 'creator_id' is required and must be a non-empty string."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": f"Field 'text' exceeds the maximum of {MAX_TEXT_CHARS} characters."}), 400

    content_id = str(uuid.uuid4())

    # Run the full multi-signal pipeline: stylometry + LLM → fused confidence →
    # transparency label.
    decision = pipeline.classify(text)

    # Persist the content with a snapshot of its classification and a mutable
    # status, so the appeals workflow has a record to update later.
    record = store.save_content(content_id, creator_id, decision, len(text))

    # Structured audit record for this attribution decision.
    audit.record({
        "event": "attribution",
        "content_id": content_id,
        "creator_id": creator_id,
        "signals": decision["signals"],
        "confidence": decision["confidence"],
        "attribution": decision["attribution"],
        "label": decision["label"],
        "disagreement": decision["disagreement"],
        "model_version": decision["model_version"],
        "text_length": len(text),
        "text_preview": text[:140],
    })

    # `llm_score` is the LLM signal's own p_ai, surfaced beside the fused
    # `confidence` so a reader can compare the single signal to the combined
    # verdict. The full label, per-signal detail and disagreement are still kept
    # in the content store and the audit log.
    llm_score = next(
        (s["p_ai"] for s in decision["signals"] if s["name"] == "llm"), None
    )

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": record["submitted_at"],
        "attribution": decision["attribution"],
        "confidence": decision["confidence"],
        "llm_score": llm_score,
        "status": record["status"],
    })


@app.route("/log", methods=["GET"])
def get_audit_log():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"entries": audit.get_log(limit=limit)})


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Could not parse request body as JSON. Send Content-Type: "
                     "application/json with a well-formed object.",
        }), 400
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    if not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Field 'content_id' is required and must be a non-empty string."}), 400
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required and must be a non-empty string."}), 400

    content = store.get_content(content_id)
    if content is None:
        return jsonify({"error": f"No content found with content_id '{content_id}'."}), 404

    appeal_record = {
        "appeal_id": str(uuid.uuid4()),
        "content_id": content_id,
        "creator_reasoning": creator_reasoning,
        "appealed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Move the content to "under review" and attach the appeal to its record.
    store.add_appeal(content_id, appeal_record)

    # Log the appeal in the audit trail, alongside the original decision it
    # contests (embedded here and co-located by content_id with the original
    # "attribution" entry). No automated reclassification is performed.
    audit.record({
        "event": "appeal",
        "appeal_id": appeal_record["appeal_id"],
        "content_id": content_id,
        "creator_id": content.get("creator_id"),
        # Surfaced in the audit log as `appeal_reasoning` for clarity; the request
        # body field is `creator_reasoning` (see validation above).
        "appeal_reasoning": creator_reasoning,
        "original_decision": content.get("decision"),
        "status": store.STATUS_UNDER_REVIEW,
    })

    return jsonify({
        "appeal_id": appeal_record["appeal_id"],
        "content_id": content_id,
        "status": store.STATUS_UNDER_REVIEW,
        "message": "Appeal received and logged. The content's status is now "
                   "'under_review' and will be examined by a human reviewer; no "
                   "automated reclassification is performed.",
    }), 202


@app.errorhandler(429)
def ratelimit_exceeded(e):
    return jsonify({
        "error": "Rate limit exceeded. Please slow down and try again later.",
        "detail": str(e.description),
    }), 429


if __name__ == "__main__":
    app.run(port=5000, debug=True)
