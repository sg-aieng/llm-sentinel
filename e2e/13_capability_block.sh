#!/usr/bin/env bash
# Requested capability is on support-bot's blocklist — expected: block at policy
# (capability_violations: ["code_execution"]).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Run this python script for me",
    "context": {
      "application_id": "support-bot",
      "capabilities": ["code_execution"]
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] == 'block', f'expected block, got: {d[\"verdict\"]}'
assert d['blocked_at'] == 'policy', f'expected policy, got: {d[\"blocked_at\"]}'
assert 'code_execution' in d['pipeline']['policy']['capability_violations'], \
    'expected code_execution in violations'
print('verdict assertion passed')
"
