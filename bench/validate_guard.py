"""Validate the batch guard: short states batch, long states run solo, and the
server stays responsive (this is what failed before the guard)."""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://192.168.1.242:8001"
LOREM = (
    "The monitoring dashboard flagged elevated error rates on the checkout service. "
    "Latency p99 rose from 240ms to 1.8s and the payment provider reported timeouts. "
    "Customer complaints arrived via support chat mentioning duplicate charges. "
)


def call(body: dict, timeout: float = 120.0) -> tuple[float, int, dict | None]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{BASE}/v1/systemone", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (time.perf_counter() - t0) * 1000, resp.status, json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return (time.perf_counter() - t0) * 1000, 0, {"error": str(exc)[:120]}


def stats() -> dict:
    with urllib.request.urlopen(f"{BASE}/stats", timeout=10) as resp:
        return json.loads(resp.read().decode())


def make(nq: int, target_tokens: int, seed: int) -> dict:
    words = int(target_tokens / 1.3)
    parts = []
    while sum(len(p.split()) for p in parts) < words:
        parts.append(LOREM + f"Reference {seed}-{len(parts)}. ")
    questions = {f"q{i}": {"type": "noul", "instructions": f"Is claim {i} plausible?"} for i in range(nq)}
    return {"model": "clef-flash", "state": " ".join(parts), "questions": questions}


def health_ok(timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def phase(label: str, bodies: list[dict], conc: int, probes: int = 8) -> dict:
    """Fire `conc` concurrent requests repeatedly; probe /health during the load."""
    health_failures = 0
    with ThreadPoolExecutor(max_workers=conc) as pool:
        for _ in range(3):
            futures = [pool.submit(call, b) for b in bodies]
            # probe /health while inference is busy
            for _ in range(probes):
                if not health_ok():
                    health_failures += 1
                time.sleep(0.3)
            results = [f.result() for f in futures]
    ok = sum(1 for _, code, _ in results if code == 200)
    s = stats()
    out = {
        "ok": f"{ok}/{len(results)}",
        "health_probes_failed_during_load": health_failures,
        "avg_batch_size": s["avg_batch_size"],
        "errors_total": s["errors"],
    }
    print(f"{label}: {out}")
    return out


def main() -> None:
    print("== guard validation ==")
    # phase 1: genuinely short states (~280 est tokens incl. questions) -> should batch
    short = [make(3, 150, 1000 + i) for i in range(8)]
    s1 = stats()
    phase("short ~150tok state x8concurrent", short, 8)
    s2 = stats()
    req_d, bat_d = s2["requests"] - s1["requests"], s2["batches"] - s1["batches"]
    short_batching = req_d > 0 and req_d / bat_d > 1.5
    print(f"  short-phase batching: {req_d} reqs in {bat_d} batches ({req_d / max(bat_d, 1):.1f} avg) -> {'PASS' if short_batching else 'FAIL'}")

    # phase 2: LONG states (the shape that wedged the box pre-guard) -> must run
    # solo AND keep /health responsive
    long_bodies = [make(8, 1000, 2000 + i) for i in range(8)]
    s2b = stats()
    phase("long 1000tok x8concurrent (was the wedge)", long_bodies, 8)
    s3 = stats()
    req_d2, bat_d2 = s3["requests"] - s2b["requests"], s3["batches"] - s2b["batches"]
    solo = req_d2 > 0 and bat_d2 / req_d2 > 0.9  # ~1 batch per request
    print(f"  long-phase: {req_d2} reqs in {bat_d2} batches -> solo {'PASS' if solo else 'FAIL'}")
    print(f"  final stats: {json.dumps(s3)}")

    # phase 3: mixed load — health probes during both
    mixed = short + long_bodies
    p3 = phase("mixed load", mixed, 16)
    alive = p3["health_probes_failed_during_load"] == 0
    print(f"  event loop responsive under load: {'PASS' if alive else 'FAIL'}")

    verdict = short_batching and solo and alive
    print("GUARD VALIDATION:", "PASS" if verdict else "FAIL")


if __name__ == "__main__":
    main()
