#!/usr/bin/env bash
# Clean model output. Expected: allow, safety.safe=true (when Llama Guard
# is loaded; review when it isn't — fail-safe, never block a clean response).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/output/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "What is the capital of France?",
    "output": "The capital of France is Paris. It has been the capital since 987 AD."
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] in ('allow', 'review'), f'expected allow/review, got: {d[\"verdict\"]}'
assert d['verdict'] != 'block', 'clean output must not be blocked'
print('verdict assertion passed')
"
