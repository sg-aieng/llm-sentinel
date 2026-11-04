#!/usr/bin/env bash
# Model response containing PII, vetted under support-bot's policy
# (pii.action: redact). Expected: allow with sanitized_output containing
# [PERSON_1] / [EMAIL_1] placeholders (when PII model loaded).
# When PII model unavailable → review (fail-safe).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/output/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "output": "The account belongs to John Smith, reachable at john.smith@example.com.",
    "context": { "application_id": "support-bot" }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] in ('allow', 'review'), f'expected allow/review, got: {d[\"verdict\"]}'
assert d['verdict'] != 'block', 'PII redaction should not block this output'
print('verdict assertion passed')
"
