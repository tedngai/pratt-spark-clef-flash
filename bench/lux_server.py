"""Minimal SystemOne-compatible server for vllm-sr/Decision-2.0-Lux-9B.

Runs inside a second container (from the clef-flash image) on host port 8002.
Same request/response shapes as the clef-flash server so run_bench.py works
unchanged: POST /v1/systemone, GET /health. Serial inference only — a lock
guards the model, matching the concurrency=1 benchmark protocol.
"""

from __future__ import annotations

import os
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get("LUX_MODEL_ID", "vllm-sr/Decision-2.0-Lux-9B")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "Decision-2.0-Lux-9B")
PORT = int(os.environ.get("PORT", "8002"))

app = FastAPI()
state = {"model": None, "ready": False, "error": None, "device": "cuda", "loaded_at": None}
infer_lock = threading.Lock()


def load_model() -> None:
    try:
        from transformers import AutoModel  # noqa: PLC0415

        # Lux's runtime fixes its own numerics (GPU: BF16 autocast, FP32 head) —
        # passing dtype explicitly raises. device_map streams weights to GPU
        # without a CPU fp32 round-trip (which OOMs alongside clef-flash).
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cuda")
        model.eval()
        state["model"] = model
        state["loaded_at"] = time.time()
        state["ready"] = True
        print(f"[lux] loaded {MODEL_ID} on cuda", flush=True)
    except Exception as exc:  # noqa: BLE001
        state["error"] = repr(exc)[:500]
        print(f"[lux] LOAD FAILED: {exc!r}", flush=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state["ready"] else ("error" if state["error"] else "loading"),
        "ready": state["ready"],
        "model": SERVED_NAME,
        "device": state["device"] if state["ready"] else None,
        "error": state["error"],
        "loaded_at": state["loaded_at"],
    }


@app.post("/v1/systemone")
async def systemone(request: Request) -> JSONResponse:
    if not state["ready"]:
        return JSONResponse({"error": "model not ready", "detail": state["error"]}, status_code=503)
    body = await request.json()
    questions = body.get("questions") or {}
    model = state["model"]
    t0 = time.perf_counter()
    try:
        with infer_lock:
            result = model.system_one(state=body.get("state"), questions=questions)
    except TypeError:
        # some versions take a single record dict
        with infer_lock:
            result = model.system_one(
                {"state": body.get("state"), "questions": questions},
            )
    latency_ms = (time.perf_counter() - t0) * 1000
    answers = result.get("answers", result) if isinstance(result, dict) else result
    return JSONResponse({
        "model": SERVED_NAME,
        "answers": answers,
        "usage": {"latency_ms": round(latency_ms, 1)},
    })


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
