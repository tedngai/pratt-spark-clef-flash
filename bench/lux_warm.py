"""Steady-state latency check for Lux-9B (after first-request compilation)."""

import json
import time
import urllib.request

body = {
    "model": "Decision-2.0-Lux-9B",
    "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults",
            },
        },
    },
}

for i in range(4):
    req = urllib.request.Request(
        "http://192.168.1.242:8002/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        json.loads(resp.read().decode())
    print(f"request {i + 1}: {(time.perf_counter() - t0) * 1000:.0f} ms")
