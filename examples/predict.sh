#!/usr/bin/env bash
# Convenience /predict request (no "model" field required).
set -euo pipefail
PORT=${PORT:-8001}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}

curl -sS -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
      "status": {
        "type": "choice",
        "instructions": "What is the invoice status?",
        "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}
      },
      "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}
    }
  }' | python3 -m json.tool
