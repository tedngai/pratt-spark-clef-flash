#!/usr/bin/env bash
# Smoke test the running Clef-Flash API.
set -euo pipefail
PORT=${PORT:-8001}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}

echo "== health =="
curl -sS "$BASE_URL/health" | python3 -m json.tool

echo "== systemone =="
response=$(curl -sS -X POST "$BASE_URL/v1/systemone" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef-flash",
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
      "status": {
        "type": "choice",
        "instructions": "What is the invoice status?",
        "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}
      },
      "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}
    }
  }')
echo "$response" | python3 -m json.tool

python3 - "$response" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
answers = data["answers"]
assert "status" in answers and "large" in answers, answers
assert answers["status"]["choice"] == "overdue", answers["status"]
assert answers["large"]["noul"] > 0.5, answers["large"]
print("OK: status=overdue, large is true")
PY
