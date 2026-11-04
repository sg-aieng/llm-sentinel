#!/usr/bin/env bash
# Health check — confirms the server is up and whether the classifier loaded.
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf "$BASE_URL/healthz")
echo "$BODY" | python3 -m json.tool
# Assert status is "ok" or "degraded" (never an HTTP error or missing field).
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d.get('status') in ('ok', 'degraded'), f'unexpected status: {d}'
print('health assertion passed')
"
