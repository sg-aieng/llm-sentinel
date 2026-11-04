#!/usr/bin/env bash
# Full request with optional context and custom thresholds.
# Context ids show up in the server's audit log lines.
# Expected: allow or review (clean prompt, context fields pass through).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Draft a polite reply to this customer email.",
    "context": {
      "application_id": "support-bot",
      "user_id": "user-42",
      "session_id": "sess-abc123"
    },
    "options": {
      "thresholds": { "medium": 0.30, "high": 0.80 }
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] in ('allow', 'review'), f'expected allow/review, got: {d[\"verdict\"]}'
print('verdict assertion passed')
"
