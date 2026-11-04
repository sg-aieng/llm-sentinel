#!/usr/bin/env bash
# Clean prompt — expected verdict: allow (or review if classifier isn't loaded).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is the capital of France?"}')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] in ('allow', 'review'), f'expected allow/review, got: {d[\"verdict\"]}'
print('verdict assertion passed')
"
