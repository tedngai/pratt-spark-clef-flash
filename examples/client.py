#!/usr/bin/env python3
"""Call the Clef-Flash API from Python, including an image input.

Usage:
    python client.py            # text-only invoice example
    python client.py --image URL  # add an image to the state
"""
from __future__ import annotations

import argparse
import json
import urllib.request

BASE_URL = "http://127.0.0.1:8001"


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional image URL to attach to the state")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    global BASE_URL  # noqa: PLW0603
    BASE_URL = args.base_url

    payload = {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {
                    "paid": "Invoice is paid.",
                    "overdue": "Invoice is past due.",
                    "draft": "Not sent.",
                },
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    }
    if args.image:
        payload["images"] = [args.image]

    print(json.dumps(post("/predict", payload), indent=2))


if __name__ == "__main__":
    main()
