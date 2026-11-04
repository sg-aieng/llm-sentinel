#!/usr/bin/env bash
# Unknown field in body — expected: HTTP 422 from request validation.
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

STATUS=$(curl -sf -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "hello", "unexpected_field": "should be rejected"}' || true)
echo "HTTP status: $STATUS"
[ "$STATUS" = "422" ] || { echo "FAIL: expected 422, got $STATUS"; exit 1; }
echo "HTTP status assertion passed"
