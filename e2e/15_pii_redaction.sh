#!/usr/bin/env bash
# Prompt contains an email and phone number — support-bot's policy action is
# redact. Expected: allow (or review when PII model unavailable), sanitized_prompt
# has [EMAIL_1]/[PHONE_1] placeholders when PII model is loaded.
# Requires the PII model to be loaded (first run downloads it).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Draft a reply to John Smith at john.smith@example.com, phone 415-555-0132, about his refund.",
    "context": {
      "application_id": "support-bot"
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
# PII model unavailable → 'review'; loaded → 'allow' with redacted prompt
assert d['verdict'] in ('allow', 'review'), f'expected allow/review, got: {d[\"verdict\"]}'
assert d['verdict'] != 'block', 'PII redaction should not block this prompt'
print('verdict assertion passed')
"
