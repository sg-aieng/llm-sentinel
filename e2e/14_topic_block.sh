#!/usr/bin/env bash
# Prompt hits support-bot's blocked medical_advice topic — expected: block at
# policy (topic_matches: ["medical_advice"]).
set -euo pipefail
BASE_URL="${SENTINEL_URL:-http://localhost:8000}"

BODY=$(curl -sf -X POST "$BASE_URL/v1/input/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "What dosage of ibuprofen should I take for a headache?",
    "context": {
      "application_id": "support-bot"
    }
  }')
echo "$BODY" | python3 -m json.tool
echo "$BODY" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['verdict'] == 'block', f'expected block, got: {d[\"verdict\"]}'
assert d['blocked_at'] == 'policy', f'expected policy, got: {d[\"blocked_at\"]}'
assert 'medical_advice' in d['pipeline']['policy']['topic_matches'], \
    'expected medical_advice in topic_matches'
print('verdict assertion passed')
"
