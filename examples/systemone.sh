#!/usr/bin/env bash
# Native Jev/SystemOne request against the Clef-Flash API.
set -euo pipefail
PORT=${PORT:-8001}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}

curl -sS -X POST "$BASE_URL/v1/systemone" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
      },
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}
    }
  }' | python3 -m json.tool
