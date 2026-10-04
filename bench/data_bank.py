"""Dataset access for the batching experiment (container or local paths)."""

from __future__ import annotations

import json
from pathlib import Path

_CANDIDATES = [
    Path("/work/bench/data/banking77.jsonl"),  # inside lux9b container
    Path(__file__).parent / "data" / "banking77.jsonl",  # local
]


def _load_all() -> list[dict]:
    for p in _CANDIDATES:
        if p.exists():
            return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    raise FileNotFoundError("banking77.jsonl not found")


def load_banking_rows(n: int) -> list[dict]:
    """First n rows with the full 77-option intent question (same encoding as run_bench)."""
    all_rows = _load_all()
    labels = sorted({r["label"] for r in all_rows})
    q = {
        "type": "choice",
        "instructions": "Which banking service does this customer message ask about? Choose the single best intent.",
        "criteria": {label: label.replace("_", " ") for label in labels},
    }
    return [{"state": r["state"], "questions": {"q": q}} for r in all_rows[:n]]


def small_criteria() -> tuple[list[str], dict]:
    """8 short states with a 3-option question — mirrors the clef stress-test body."""
    states = [
        f"Customer message {i}: the card payment failed at checkout and the amount is still "
        f"reserved on the account. The customer completed the purchase twice and needs help now."
        for i in range(8)
    ]
    q = {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoices, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    }
    return states, {"q": q}
