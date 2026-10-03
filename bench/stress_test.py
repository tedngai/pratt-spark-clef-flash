"""Production-readiness stress test for a SystemOne-compatible endpoint.

Phases (all against the same small-schema profile so results are comparable):
  1. baseline      — serial latency (c=1) reference
  2. concurrency   — c=4/8/16: does aggregate throughput scale? (stock server: no;
                     hardened server with batching: should scale to ~BATCH_MAX x)
  3. burst         — 100 requests at once: latency collapse, 500s, survival
  4. correctness   — batched answers must match serial answers exactly

Writes results/stress.json and prints a verdict table.

Usage:
  python stress_test.py [--base-url http://192.168.1.242:8001] [--label stock]
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q / 100 * len(s)))] if s else 0.0


def make_request(i: int) -> dict:
    return {
        "model": "clef-flash",
        "state": f"Support ticket {i}: customer reports the checkout service returns 502s "
                 f"since this morning and orders are blocked. Reference {1000 + i}.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "Purchases"},
            },
            "urgent": {"type": "noul", "instructions": "Does this need immediate attention?"},
        },
    }


def one_call(base_url: str, body: dict, timeout: float = 120.0) -> tuple[float, int, dict | None, str]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/systemone", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
        return (time.perf_counter() - t0) * 1000, resp.status, result, ""
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode()[:120]
        except Exception:  # noqa: BLE001
            detail = ""
        return (time.perf_counter() - t0) * 1000, exc.code, None, detail
    except Exception as exc:  # noqa: BLE001
        return (time.perf_counter() - t0) * 1000, 0, None, str(exc)[:120]


def phase_baseline(base_url: str, n: int = 15) -> dict:
    lats = []
    for i in range(n):
        ms, status, _, _ = one_call(base_url, make_request(i))
        if status != 200:
            return {"error": f"baseline request failed: {status}"}
        lats.append(ms)
    return {"n": n, "p50_ms": round(percentile(lats, 50), 1),
            "p95_ms": round(percentile(lats, 95), 1),
            "throughput_rps": round(1000 / statistics.mean(lats), 2)}


def phase_concurrency(base_url: str, levels: list[int], per_level: int = 32) -> dict:
    out = {}
    for c in levels:
        # warm this level's (batch, seqlen) shapes: Triton autotunes on first
        # encounter (~seconds) and we want steady-state numbers, not that cost
        warm = [make_request(9000 + i) for i in range(c)]
        with ThreadPoolExecutor(max_workers=c) as pool:
            list(pool.map(lambda b: one_call(base_url, b), warm))
        bodies = [make_request(i) for i in range(per_level)]
        t0 = time.perf_counter()
        codes: list[int] = []
        with ThreadPoolExecutor(max_workers=c) as pool:
            futures = [pool.submit(one_call, base_url, b) for b in bodies]
            results = [f.result() for f in futures]
        wall = time.perf_counter() - t0
        codes = [r[1] for r in results]
        ok = codes.count(200)
        errs = {code: codes.count(code) for code in set(codes) if code != 200}
        out[f"c={c}"] = {
            "n": per_level, "ok": ok, "errors": errs or None,
            "wall_s": round(wall, 1),
            "throughput_rps": round(per_level / wall, 2),
            "client_p50_ms": round(percentile([r[0] for r in results if r[1] == 200], 50), 1),
            "client_p95_ms": round(percentile([r[0] for r in results if r[1] == 200], 95), 1),
        }
        print(f"  c={c}: {out[f'c={c}']['throughput_rps']} req/s, p50 {out[f'c={c}']['client_p50_ms']}ms, errors={errs or 'none'}")
        time.sleep(1.0)  # let the queue drain between levels
    return out


def phase_burst(base_url: str, n: int = 100) -> dict:
    """All n requests hit at once (open-loop). Measures shedding/survival."""
    barrier = threading.Barrier(n)
    results: list[tuple] = [None] * n

    def worker(i: int) -> None:
        body = make_request(200 + i)
        barrier.wait()
        results[i] = one_call(base_url, body, timeout=180.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    codes = [r[1] for r in results]
    summary = {
        "n": n, "wall_s": round(wall, 1),
        "ok": codes.count(200),
        "errors": {code: codes.count(code) for code in set(codes) if code != 200},
        "ok_p50_ms": round(percentile([r[0] for r in results if r[1] == 200], 50), 1),
        "ok_p95_ms": round(percentile([r[0] for r in results if r[1] == 200], 95), 1),
        "ok_max_ms": round(max((r[0] for r in results if r[1] == 200), default=0), 1),
    }
    print(f"  burst {n}: ok={summary['ok']}, errors={summary['errors']}, wall={summary['wall_s']}s")
    # survival check
    ms, status, _, _ = one_call(base_url, make_request(999))
    summary["survives_after"] = status == 200
    summary["post_burst_latency_ms"] = round(ms, 1)
    return summary


def phase_correctness(base_url: str, n: int = 12) -> dict:
    """Serial reference vs batched-under-concurrency: choices must match; bf16
    batch-composition jitter on probabilities must stay below ~3e-3."""
    bodies = [make_request(300 + i) for i in range(n)]
    serial = {}
    for b in bodies:
        _, _, result, _ = one_call(base_url, b)
        assert result is not None
        serial[json.dumps(b, sort_keys=True)] = result["answers"]

    def answers(b: dict):
        _, status, result, _ = one_call(base_url, b)
        return result["answers"] if status == 200 and result else None

    with ThreadPoolExecutor(max_workers=n) as pool:
        batched = list(pool.map(answers, bodies))

    choice_mismatch = 0
    missing = 0
    max_delta = 0.0
    for b, answer in zip(bodies, batched):
        key = json.dumps(b, sort_keys=True)
        if answer is None:
            missing += 1
            continue
        ref = serial[key]
        for q in ref:
            if answer[q].get("choice") != ref[q].get("choice"):
                choice_mismatch += 1
            probs_a, probs_b = answer[q].get("probabilities") or {}, ref[q].get("probabilities") or {}
            for opt in set(probs_a) | set(probs_b):
                max_delta = max(max_delta, abs(probs_a.get(opt, 0) - probs_b.get(opt, 0)))
            if "noul" in answer[q]:
                max_delta = max(max_delta, abs(answer[q]["noul"] - ref[q]["noul"]))
    return {
        "n": n,
        "missing": missing,
        "choice_mismatches": choice_mismatch,
        "max_prob_delta": round(max_delta, 5),
        "note": "bf16 batch-composition jitter; decisions must match, probabilities <3e-3",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://192.168.1.242:8001")
    ap.add_argument("--label", default="stock")
    ap.add_argument("--skip-burst", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as resp:
        health = json.loads(resp.read().decode())
    if not health.get("ready"):
        raise SystemExit(f"endpoint not ready: {health}")

    print(f"== stress test against {args.base_url} ({args.label}) ==")
    out: dict = {"label": args.label, "base_url": args.base_url}

    print("phase 1: baseline (serial)")
    out["baseline"] = phase_baseline(args.base_url)
    print(f"  p50 {out['baseline'].get('p50_ms')}ms, {out['baseline'].get('throughput_rps')} req/s")

    print("phase 2: concurrency scaling")
    out["concurrency"] = phase_concurrency(args.base_url, [4, 8, 16])

    if not args.skip_burst:
        print("phase 3: burst (100 at once)")
        out["burst"] = phase_burst(args.base_url)

    print("phase 4: correctness (batched vs serial)")
    out["correctness"] = phase_correctness(args.base_url)
    c = out["correctness"]
    print(f"  choice mismatches: {c['choice_mismatches']}, max prob delta: {c['max_prob_delta']}")

    path = RESULTS / f"stress_{args.label}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
