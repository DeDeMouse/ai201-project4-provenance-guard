"""Robust submission client for Provenance Guard.

Sends text to the /submit endpoint as correctly-encoded JSON. The point is to keep
the shell out of JSON construction: apostrophes, quotes and em-dashes in the text
routinely corrupt a hand-built `curl -d '{...}'` payload before it ever leaves the
machine (an apostrophe ends a single-quoted shell string), which is why such
submissions appear to "never reach the server". Reading the text from a file or
stdin sidesteps shell quoting entirely.

Uses only the standard library (urllib) so it adds no dependency.

Usage:
    python submit.py --creator alice --file story.txt
    python submit.py --creator alice --text "I've been thinking — really."
    echo "ok so i finally tried that ramen place... won't go back." \\
        | python submit.py --creator alice
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def submit(url: str, text: str, creator_id: str, timeout: float = 30.0) -> tuple:
    """POST {text, creator_id} as JSON; return (status_code, body_text)."""
    payload = json.dumps({"text": text, "creator_id": creator_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a JSON body
        return exc.code, exc.read().decode("utf-8")


def _read_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    if args.text is not None:
        return args.text
    return sys.stdin.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit text to Provenance Guard /submit.")
    ap.add_argument("--url", default="http://127.0.0.1:5000/submit",
                    help="submission endpoint (default: %(default)s)")
    ap.add_argument("--creator", required=True, help="creator_id to attribute the submission to")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", help="read submission text from this file")
    src.add_argument("--text", help="submission text as a single argument")
    args = ap.parse_args()

    text = _read_text(args)
    if not text.strip():
        print("No text to submit (empty --file/--text/stdin).", file=sys.stderr)
        return 2

    status, body = submit(args.url, text, args.creator)
    print(f"HTTP {status}")
    try:
        print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
