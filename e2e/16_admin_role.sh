#!/usr/bin/env bash
# Same code_execution request as 13, but user-42 has the admin role
# (identities.yaml), and support-bot's policy grants admins code_execution.
# Expected: allow or review (clean prompt; admin overrides capability block).
# Requires identities.yaml (copy identities.example.yaml).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Run this python script for me",
    "context": {
      "application_id": "support-bot",
      "user_id": "user-42",
      "capabilities": ["code_execution"]
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] in ('allow', 'review'), \
    f'expected allow/review (admin role overrides capability block), got: {d[\"verdict\"]}'
print('verdict assertion passed')
"
