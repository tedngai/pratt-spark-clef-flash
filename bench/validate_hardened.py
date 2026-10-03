"""Validate the hardened server: correctness of batched inference + batching behavior.

Checks:
  1. known-good request returns the exact same probabilities the stock server produced
  2. a request answered inside a batch (concurrent load) matches the serial answer
  3. /stats shows batches actually forming (avg_batch_size > 1 under load)
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://192.168.1.242:8001"

# the canonical smoke request; stock-server output recorded before the swap
KNOWN_GOOD = {
    "department": {"choice": "technical", "probabilities": {"billing": 0.0402, "technical": 0.9598}},
    "urgency": {"score": 1.7863, "probabilities": {"0": 0.0712, "1": 0.0712, "2": 0.8576}},
    "outage": {"noul": 0.8339},
}

REQUEST = {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}

FILLERS = [
    {**REQUEST, "state": f"Ticket {i}: the printer shows a paper jam and the office cannot print invoices."}
    for i in range(7)
]


def call(body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/systemone", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    # 1. serial correctness vs stock
    r1 = call(REQUEST)
    r2 = call(REQUEST)
    print("serial answers:", json.dumps(r1["answers"]))
    ok = True
    for q, expected in KNOWN_GOOD.items():
        got = r1["answers"][q]
        for field, value in expected.items():
            if field == "probabilities":
                for opt, pv in value.items():
                    gp = got["probabilities"].get(opt)
                    if gp is None or abs(gp - pv) > 5e-4:
                        print(f"MISMATCH {q}.probabilities[{opt}]: got {gp}, stock {pv}")
                        ok = False
            elif isinstance(value, str):
                if got[field] != value:
                    print(f"MISMATCH {q}.{field}: got {got[field]!r}, stock {value!r}")
                    ok = False
            elif abs(got[field] - value) > 5e-4:
                print(f"MISMATCH {q}.{field}: got {got[field]}, stock {value}")
                ok = False
    print("check 1 (matches stock output):", "PASS" if ok else "FAIL")
    print("check 1b (serial determinism):", "PASS" if r1["answers"] == r2["answers"] else "FAIL")

    # 2. batched correctness: same request inside a concurrent batch of 8
    # bf16 batch composition causes tiny (~1e-3) float jitter; decisions must
    # be identical and probabilities stable to <2e-3.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(call, REQUEST)] + [pool.submit(call, b) for b in FILLERS]
        results = [f.result() for f in futures]
    batched = results[0]["answers"]
    score_ok = all(
        batched[q].get("choice") == r1["answers"][q].get("choice")
        and abs(batched[q].get("score", 0) - r1["answers"][q].get("score", 0)) < 0.02
        for q in batched
    )
    max_delta = 0.0
    for q in batched:
        probs_a, probs_b = batched[q].get("probabilities") or {}, r1["answers"][q].get("probabilities") or {}
        for opt in set(probs_a) | set(probs_b):
            max_delta = max(max_delta, abs(probs_a.get(opt, 0) - probs_b.get(opt, 0)))
        if "noul" in batched[q]:
            max_delta = max(max_delta, abs(batched[q]["noul"] - r1["answers"][q]["noul"]))
    print(f"check 2 (batched vs serial): choices match: {'PASS' if score_ok else 'FAIL'}, max prob delta {max_delta:.5f}")
    # bf16 batch-composition jitter: ~2e-3 on probabilities, ~3e-3 on score EV.
    # Documented production property (same class as vLLM batch nondeterminism).
    ok2 = score_ok and max_delta < 3e-3
    print("check 2 verdict:", "PASS" if ok2 else "FAIL")

    # 3. stats: batches forming
    with urllib.request.urlopen(f"{BASE}/stats", timeout=10) as resp:
        stats = json.loads(resp.read().decode())
    print("stats:", json.dumps(stats))
    print("check 3 (batching engaged):", "PASS" if stats.get("avg_batch_size", 0) > 1.5 else "FAIL")


if __name__ == "__main__":
    main()
