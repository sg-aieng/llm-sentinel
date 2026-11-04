"""Structured JSON audit logging to stdout.

Never logs prompt content — only the SHA-256 hash.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def audit_log(
    request_id: str,
    stage: str,
    verdict: str,
    hashed_prompt: str,
    application_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    risk_score: float | None = None,
    violations: list[str] | None = None,
) -> None:
    record: dict = {
        "ts": utc_now(),
        "request_id": request_id,
        "stage": stage,
        "verdict": verdict,
        "prompt_hash": hashed_prompt,
    }
    if application_id is not None:
        record["application_id"] = application_id
    if user_id is not None:
        record["user_id"] = user_id
    if session_id is not None:
        record["session_id"] = session_id
    if risk_score is not None:
        record["risk_score"] = risk_score
    if violations:
        record["violations"] = violations
    print(json.dumps(record, separators=(",", ":")), file=sys.stdout, flush=True)
