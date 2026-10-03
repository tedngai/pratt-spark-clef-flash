"""Run decision benchmarks (BANKING77, CLINC150+OOS, ANLI) through a SystemOne-compatible endpoint.

Results are appended to results/<dataset>.jsonl (one record per sample) and are resumable:
already-recorded sample ids are skipped on re-run.

Usage:
  python run_bench.py --dataset banking77 --concurrency 4
  python run_bench.py --dataset all --concurrency 4 --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Bearer token appended to every request when set (e.g. OpenRouter API key).
AUTH_TOKEN: str | None = None


# --------------------------------------------------------------------------- #
# Question encoders: dataset row -> SystemOne request questions
# --------------------------------------------------------------------------- #

def readable(label: str) -> str:
    return label.replace("_", " ").strip()


def questions_for(dataset: str, row: dict) -> dict:
    """Build the questions schema for a dataset row. Returns the questions dict."""
    if dataset == "banking77":
        labels = sorted({r["label"] for r in load_dataset("banking77")})
        return {
            "intent": {
                "type": "choice",
                "instructions": "Which banking service does this customer message ask about? Choose the single best intent.",
                "criteria": {label: readable(label) for label in labels},
            }
        }
    if dataset == "clinc150":
        labels = sorted({r["label"] for r in load_dataset("clinc150")})
        criteria = {}
        for label in labels:
            if label == "oos":
                criteria[label] = "Out of scope: none of the listed intents apply"
            else:
                criteria[label] = readable(label)
        return {
            "intent": {
                "type": "choice",
                "instructions": "Classify the user utterance into exactly one of the listed intents, or out-of-scope if none apply.",
                "criteria": criteria,
            }
        }
    if dataset == "anli":
        return {
            "relationship": {
                "type": "choice",
                "instructions": "Given the premise, what is the relationship to the hypothesis?",
                "criteria": {
                    "entailment": "The premise entails the hypothesis: the hypothesis is definitely true given the premise",
                    "neutral": "The hypothesis is neither entailed nor contradicted by the premise",
                    "contradiction": "The premise contradicts the hypothesis: the hypothesis is definitely false given the premise",
                },
            }
        }
    raise ValueError(f"unknown dataset {dataset}")


def extract_answer(dataset: str, response: dict) -> dict:
    """Normalize a SystemOne response into predicted label + probabilities + confidence."""
    if dataset in ("banking77", "clinc150"):
        a = response["answers"]["intent"]
        return {"predicted": a["choice"], "probabilities": a["probabilities"], "confidence": a.get("confidence")}
    if dataset == "anli":
        a = response["answers"]["relationship"]
        return {"predicted": a["choice"], "probabilities": a["probabilities"], "confidence": a.get("confidence")}
    raise ValueError(f"unknown dataset {dataset}")


# --------------------------------------------------------------------------- #
# Data + result IO
# --------------------------------------------------------------------------- #

_data_cache: dict[str, list[dict]] = {}
_cache_lock = threading.Lock()


def load_dataset(name: str) -> list[dict]:
    with _cache_lock:
        if name not in _data_cache:
            path = DATA / f"{name}.jsonl"
            _data_cache[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return _data_cache[name]


def load_done_ids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def call_systemone(base_url: str, endpoint: str, model: str, state, questions: dict,
                   retries: int = 3, timeout: float = 120.0) -> tuple[dict, float, float]:
    """POST one decision. Returns (response_json, client_ms, server_ms)."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        headers["X-OpenRouter-Title"] = "clef-benchmark"
    last_exc: Exception | None = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{base_url}{endpoint}", data=body,
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            client_ms = (time.perf_counter() - t0) * 1000
            server_ms = result.get("usage", {}).get("latency_ms", 0.0)
            return result, client_ms, server_ms
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                pass
            last_exc = exc
            wait = min(2 ** attempt, 10)
            log(f"    request failed (attempt {attempt + 1}/{retries}): {exc} body={body}; retry in {wait}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            wait = min(2 ** attempt, 10)
            log(f"    request failed (attempt {attempt + 1}/{retries}): {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"request failed after {retries} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_dataset(dataset: str, base_url: str, endpoint: str, model: str,
                concurrency: int, limit: int | None, tag: str = "") -> None:
    rows = load_dataset(dataset)
    out_path = RESULTS / f"{tag}{dataset}.jsonl"
    done = load_done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    total_done, total = len(done), len(rows)
    log(f"[{dataset}] {total} samples total, {total_done} already done, {len(todo)} to run (concurrency={concurrency})")
    if not todo:
        return

    # build questions once per dataset (schema is shared across rows)
    questions = questions_for(dataset, {"id": "probe", "label": ""})

    write_lock = threading.Lock()
    progress_every = max(1, len(todo) // 20)
    started = time.perf_counter()
    done_in_run = 0

    def work(row: dict) -> None:
        nonlocal done_in_run
        result, client_ms, server_ms = call_systemone(base_url, endpoint, model, row["state"], questions)
        answer = extract_answer(dataset, result)
        record = {
            "id": row["id"],
            "expected": row["label"],
            **answer,
            "correct": answer["predicted"] == row["label"],
            "client_ms": round(client_ms, 2),
            "server_ms": server_ms,
            "input_tokens": result.get("usage", {}).get("input_tokens"),
            "meta": row.get("meta", {}),
        }
        with write_lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_in_run += 1
            n = done_in_run
        if n % progress_every == 0:
            elapsed = time.perf_counter() - started
            log(f"[{dataset}] {n}/{len(todo)} done | {n / elapsed:.2f} req/s | {elapsed:.0f}s elapsed")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, row) for row in todo]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                log(f"[{dataset}] SAMPLE FAILED: {exc}")

    elapsed = time.perf_counter() - started
    log(f"[{dataset}] finished: {len(todo)} samples in {elapsed:.0f}s ({len(todo) / elapsed:.2f} req/s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150", "anli", "all"])
    ap.add_argument("--base-url", default="http://192.168.1.242:8001")
    ap.add_argument("--endpoint", default="/v1/systemone",
                    help="API path, e.g. /v1/systemone (local) or /api/alpha/decisions (OpenRouter)")
    ap.add_argument("--model", default="clef-flash")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap new samples per dataset (smoke tests)")
    ap.add_argument("--tag", default="", help="prefix for result files, e.g. lux9b_ -> results/lux9b_banking77.jsonl")
    ap.add_argument("--auth-env", default=None,
                    help="env var name holding a bearer token, e.g. OPENROUTER_API_KEY")
    ap.add_argument("--skip-health", action="store_true", help="skip the /health probe (cloud APIs)")
    args = ap.parse_args()

    global AUTH_TOKEN
    if args.auth_env:
        AUTH_TOKEN = os.environ[args.auth_env]

    # health check
    if args.skip_health:
        log("skipping health check")
    else:
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode())
        if not health.get("ready"):
            sys.exit(f"endpoint not ready: {health}")
        log(f"endpoint ready: {health.get('model')} on {health.get('device')}")

    datasets = ["banking77", "clinc150", "anli"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, args.base_url, args.endpoint, args.model, args.concurrency, args.limit, args.tag)
    log("all done")


if __name__ == "__main__":
    main()
