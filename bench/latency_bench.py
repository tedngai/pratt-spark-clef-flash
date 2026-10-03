"""Latency microbenchmark for a SystemOne-compatible endpoint.

Profiles:
  A. state-size scaling   — 1 choice question (5 options), states of ~100/1k/4k/8k tokens
  B. question-count       — 1/8/32/64 noul questions, ~300-token state
  C. image                — 1024x1024 PNG, 1 choice question (4 options)
  D. concurrency          — state ~1k tokens, 8 questions, at c=1/2/4/8 (aggregate throughput)

Serial profiles (A/B/C) record per-request latency; profile D records aggregate throughput.
Results -> results/latency.json

Usage:
  python latency_bench.py [--base-url http://192.168.1.242:8001]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LOREM = (
    "The monitoring dashboard flagged elevated error rates on the checkout service. "
    "Latency p99 rose from 240ms to 1.8s and the payment provider reported timeouts. "
    "Customer complaints arrived via support chat mentioning duplicate charges. "
)


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    idx = min(len(s) - 1, int(q / 100 * len(s)))
    return s[idx]


def call(base_url: str, body: dict, timeout: float = 300.0) -> tuple[float, float, dict]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/systemone", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    client_ms = (time.perf_counter() - t0) * 1000
    return client_ms, result.get("usage", {}).get("latency_ms", 0.0), result


def make_state(target_tokens: int, rng: random.Random) -> str:
    """Approximate token count with ~1.3 tokens per word."""
    words_needed = int(target_tokens / 1.3)
    parts: list[str] = []
    while sum(len(p.split()) for p in parts) < words_needed:
        parts.append(LOREM + f"Incident report ID {rng.randint(1000, 9999)} at {rng.random():.5f}.")
    return " ".join(parts)[: None]


def profile_a(base_url: str, reps: int, rng: random.Random) -> dict:
    out = {}
    for target in (100, 1000, 4000, 8000):
        state = make_state(target, rng)
        body = {
            "model": "clef-flash",
            "state": state,
            "questions": {
                "severity": {
                    "type": "choice",
                    "instructions": "How severe is this incident?",
                    "criteria": {"low": "cosmetic", "medium": "degraded", "high": "major outage", "critical": "data loss", "unknown": "cannot determine"},
                }
            },
        }
        # warmup one
        call(base_url, body)
        clients, servers = [], []
        for _ in range(reps):
            c, s, _ = call(base_url, body)
            clients.append(c)
            servers.append(s)
        out[f"state~{target}tok"] = {
            "n": reps,
            "client_p50_ms": round(percentile(clients, 50), 1),
            "client_p95_ms": round(percentile(clients, 95), 1),
            "server_p50_ms": round(percentile(servers, 50), 1),
            "server_p95_ms": round(percentile(servers, 95), 1),
        }
        print(f"  A state~{target}tok: server p50 {out[f'state~{target}tok']['server_p50_ms']}ms")
    return out


def profile_b(base_url: str, reps: int, rng: random.Random) -> dict:
    out = {}
    state = make_state(300, rng)
    for nq in (1, 8, 32, 64):
        questions = {
            f"q{i}": {"type": "noul", "instructions": f"Is claim {i} plausible given the incident notes?"}
            for i in range(nq)
        }
        body = {"model": "clef-flash", "state": state, "questions": questions}
        call(base_url, body)  # warmup
        clients, servers = [], []
        for _ in range(reps):
            c, s, _ = call(base_url, body)
            clients.append(c)
            servers.append(s)
        out[f"{nq}q"] = {
            "n": reps,
            "client_p50_ms": round(percentile(clients, 50), 1),
            "server_p50_ms": round(percentile(servers, 50), 1),
        }
        print(f"  B {nq}q: server p50 {out[f'{nq}q']['server_p50_ms']}ms")
    return out


def profile_c(base_url: str, reps: int) -> dict:
    # deterministic 1024x1024 PNG (no PIL dependency): uncompressed-ish BMP-like via zlib? Use PIL if present.
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (1024, 1024))
        draw = ImageDraw.Draw(img)
        for x in range(0, 1024, 64):
            for y in range(0, 1024, 64):
                if (x // 64 + y // 64) % 2 == 0:
                    draw.rectangle([x, y, x + 63, y + 63], fill=(200, 60, 60))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        print("  PIL unavailable; skipping image profile")
        return {}
    body = {
        "model": "clef-flash",
        "state": "Decide whether this image shows an outage status page.",
        "images": [data_url],
        "questions": {
            "screenshot_type": {
                "type": "choice",
                "instructions": "What does this image show?",
                "criteria": {"status_page": "dashboard", "invoice": "document", "photo": "photograph", "other": "something else"},
            }
        },
    }
    call(base_url, body)  # warmup (decoder + vision encoder)
    clients, servers = [], []
    for _ in range(reps):
        c, s, _ = call(base_url, body)
        clients.append(c)
        servers.append(s)
    out = {
        "image_1024": {
            "n": reps,
            "client_p50_ms": round(percentile(clients, 50), 1),
            "server_p50_ms": round(percentile(servers, 50), 1),
        }
    }
    print(f"  C image: server p50 {out['image_1024']['server_p50_ms']}ms")
    return out


def profile_d(base_url: str, reps: int, rng: random.Random) -> dict:
    """Concurrency scaling on the hot-path profile: short state (~300 tokens) + 8
    questions. Long states are intentionally NOT batched (autotune hazard on
    GB10 — see server_hardened.py SOLO_TOKEN_LIMIT), so a long-state profile
    would measure nothing but serial behavior."""
    out = {}
    state = make_state(300, rng)
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Is claim {i} plausible given the incident notes?"}
        for i in range(8)
    }
    body = {"model": "clef-flash", "state": state, "questions": questions}
    for conc in (1, 2, 4, 8):
        call(base_url, body)  # warmup

        def work(_):
            return call(base_url, body)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(work, range(reps * conc)))
        elapsed = time.perf_counter() - t0
        total = reps * conc
        out[f"c={conc}"] = {
            "n": total,
            "wall_s": round(elapsed, 1),
            "throughput_rps": round(total / elapsed, 2),
        }
        print(f"  D c={conc}: {out[f'c={conc}']['throughput_rps']} req/s aggregate")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://192.168.1.242:8001")
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    rng = random.Random(42)

    with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as resp:
        health = json.loads(resp.read().decode())
    if not health.get("ready"):
        raise SystemExit(f"endpoint not ready: {health}")

    print("profile A: state-size scaling")
    a = profile_a(args.base_url, args.reps, rng)
    print("profile B: question-count scaling")
    b = profile_b(args.base_url, args.reps, rng)
    print("profile C: image")
    c = profile_c(args.base_url, max(8, args.reps // 2))
    print("profile D: concurrency scaling")
    d = profile_d(args.base_url, 10, rng)

    out = {"profile_a_state_size": a, "profile_b_question_count": b, "profile_c_image": c, "profile_d_concurrency": d}
    path = RESULTS / "latency.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
