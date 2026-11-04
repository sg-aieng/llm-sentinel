#!/usr/bin/env bash
# Run every e2e script in order against a running server.
# Usage: ./run_all.sh            (defaults to http://localhost:8000)
#        SENTINEL_URL=http://host:port ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PASS=0
FAIL=0

for script in [0-9][0-9]_*.sh; do
  echo
  echo "═══════════════════════════════════════════"
  echo "▶ $script"
  echo "═══════════════════════════════════════════"
  if bash "$script"; then
    PASS=$((PASS + 1))
  else
    echo "✗ $script FAILED" >&2
    FAIL=$((FAIL + 1))
  fi
done

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
