#!/usr/bin/env bash
# Demonstrates the review verdict: thresholds are lowered so a prompt whose
# score falls between medium and high lands in the review band — the caller
# decides what to do with it. Expected: verdict review, sanitized_prompt present.
# When classifier is not loaded, verdict is also "review" (fail-safe).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "What is the capital of France?",
    "options": {
      "thresholds": { "medium": 0.0001, "high": 0.5 }
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] == 'review', f'expected review, got: {d[\"verdict\"]}'
print('verdict assertion passed')
"
