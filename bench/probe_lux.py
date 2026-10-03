"""Smoke test: verify Lux-9B's /v1/systemone response shape matches run_bench's parser."""

import json
import urllib.request

body = {
    "model": "Decision-2.0-Lux-9B",
    "state": "I lost my card and need to freeze it before someone uses it at a store.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Which banking service does this customer message ask about? Choose the single best intent.",
            "criteria": {
                "card_arrival": "When the card will arrive",
                "lost_or_stolen_card": "Card is lost or stolen and needs to be blocked",
                "exchange_rate": "Currency exchange rates",
            },
        },
    },
}

req = urllib.request.Request(
    "http://192.168.1.242:8002/v1/systemone",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read().decode())

print(json.dumps(result, indent=2)[:1200])
a = result["answers"]["intent"]
print("\nkeys:", sorted(a.keys()))
print("has choice:", "choice" in a, "| has probabilities:", "probabilities" in a, "| has confidence:", "confidence" in a)
