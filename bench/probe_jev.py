"""Probe OpenRouter's decisions API: response shape + latency, key never printed."""

import json
import time
import urllib.request
from pathlib import Path

env = {}
for line in Path(r"E:\LinkedinAnalysis\.env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("'\"")

key = env["OPENROUTER_API_KEY"]
print(f"key loaded: {key[:7]}...{key[-4:]} ({len(key)} chars)")

body = {
    "model": "typesafe/jev-1.13",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}

req = urllib.request.Request(
    "https://openrouter.ai/api/alpha/decisions",
    data=json.dumps(body).encode(),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "clef-benchmark",
    },
    method="POST",
)
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    ms = (time.perf_counter() - t0) * 1000
    print(f"HTTP {resp.status} in {ms:.0f} ms")
    print(json.dumps(result, indent=2)[:2000])
    a = result["answers"]["department"]
    print("\nchoice keys:", sorted(a.keys()))
except urllib.error.HTTPError as exc:
    print(f"HTTP {exc.code}: {exc.read().decode()[:500]}")
