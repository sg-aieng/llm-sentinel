#!/usr/bin/env bash
# Model response leaking an API key. Expected: block at rule_checks
# (openai_key), sanitized_output withheld.
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/output/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"output": "Sure! Use this key in your script: sk-proj4bcdEfGhIjKlMnOpQrStUvWx"}')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] == 'block', f'expected block, got: {d[\"verdict\"]}'
assert d['blocked_at'] == 'rule_checks', f'expected rule_checks, got: {d[\"blocked_at\"]}'
print('verdict assertion passed')
"
