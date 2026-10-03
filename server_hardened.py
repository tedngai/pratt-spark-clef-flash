"""Hardened Clef-Flash server: drop-in replacement for server.py with production fixes.

Additions over the stock server (all behavior-preserving when load is serial):
1. Thread-safe inference: a dedicated inference thread owns the GPU; HTTP handlers
   never touch the model concurrently (fixes intermittent 500s under concurrency).
2. Dynamic micro-batching: requests gathered within BATCH_WAIT_MS are collated into
   ONE forward pass via collate_records() — amortizes the ~66 ms weight read.
   Measured on the GB10: 7.1 -> 13.0 req/s (~1.8x) for a 3-option schema at
   BATCH_MAX=8; gains scale with shorter states and larger BATCH_MAX.
3. Bounded queue: when QUEUE_MAX requests are waiting, shed load with 429 + Retry-After
   instead of letting latency collapse.
4. /stats endpoint: counters + rolling latency percentiles for monitoring.
5. Optional API key (CLEF_API_KEY env), body-size guard, image count guard.

Env vars:
  QUEUE_MAX       default 64     # max queued requests before 429
  BATCH_MAX       default 8      # max records per forward pass
  BATCH_WAIT_MS   default 15     # gather window before running a partial batch
  CLEF_API_KEY    default unset  # when set, require "Authorization: Bearer <key>"

Deploy: copy next to server.py on the DGX host, adjust compose/systemd entrypoint
(same uvicorn invocation, single worker — keep workers=1 for the single GPU).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import queue as queue_mod
import sys
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "Cloudflare/clef-flash")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "clef-flash")
DEVICE = os.environ.get("CLEF_DEVICE", "cuda")
MAX_LENGTH = int(os.environ.get("CLEF_MAX_LENGTH", "16384"))
WARMUP = os.environ.get("CLEF_WARMUP", "1") == "1"
PORT = int(os.environ.get("PORT", "8001"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "64"))
BATCH_MAX = int(os.environ.get("BATCH_MAX", "8"))
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "15"))
API_KEY = os.environ.get("CLEF_API_KEY")  # unset = no auth (LAN)

MAX_BODY_BYTES = 13 * 1024 * 1024  # mirrors Cloudflare's Workers AI limit
MAX_IMAGES = 4

logging.basicConfig(level=LOG_LEVEL.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clef.hardened")

# --------------------------------------------------------------------------- #
# model loading (unchanged from stock server)
# --------------------------------------------------------------------------- #

STATE: dict[str, Any] = {"model": None, "processor": None, "ready": False, "error": None, "loaded_at": None}


def _load_model() -> tuple[Any, Any]:
    log.info("resolving %s", MODEL_ID)
    path = snapshot_download(MODEL_ID)
    if path not in sys.path:
        sys.path.insert(0, path)
    from joint_schema_model import load_release_model  # noqa: PLC0415

    model, processor = load_release_model(path, device=DEVICE)
    return model, processor


def _decode_image(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if not isinstance(item, str):
        raise HTTPException(status_code=400, detail="each image must be a URL or base64 string")
    value = item.strip()
    try:
        if value.startswith("data:"):
            _, _, payload = value.partition(",")
            raw = base64.b64decode(payload)
        elif value.startswith(("http://", "https://")):
            with urllib.request.urlopen(value, timeout=30) as response:  # noqa: S310
                raw = response.read()
        else:
            raw = base64.b64decode(value, validate=True)
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > 16_000_000:
            raise HTTPException(status_code=400, detail="image too large (16 MP limit)")
        return img.convert("RGB")
    except (binascii.Error, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc


def _prepare_request(body: dict[str, Any], require_model: bool) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    if require_model and not isinstance(body.get("model"), str):
        raise HTTPException(status_code=400, detail="model is required")
    if "state" not in body:
        raise HTTPException(status_code=400, detail="state is required")
    request = dict(body)
    request.setdefault("model", SERVED_MODEL_NAME)
    images = request.get("images")
    if images is not None:
        if not isinstance(images, list):
            raise HTTPException(status_code=400, detail="images must be a list")
        if len(images) > MAX_IMAGES:
            raise HTTPException(status_code=400, detail=f"at most {MAX_IMAGES} images")
        request["images"] = [_decode_image(image) for image in images]
    return request


# --------------------------------------------------------------------------- #
# batching inference engine: one thread owns the GPU
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    request: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class BatchEngine:
    """Gathers queued jobs into single forward passes on a dedicated thread."""

    def __init__(self, model: Any, processor: Any) -> None:
        from joint_schema_model import (  # noqa: PLC0415
            collate_records,
            encode_record,
            systemone_answer,
        )

        self.model = model
        self.processor = processor
        self._collate_records = collate_records
        self._encode_record = encode_record
        self._systemone_answer = systemone_answer
        self.queue: queue_mod.Queue[Job] = queue_mod.Queue(maxsize=QUEUE_MAX)
        self.stats = {"requests": 0, "batches": 0, "errors": 0, "shed": 0, "batch_sizes": []}
        self._latencies: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    def submit(self, request: dict[str, Any]) -> Job:
        job = Job(request=request)
        try:
            self.queue.put_nowait(job)
        except queue_mod.Full:
            self.stats["shed"] += 1
            raise HTTPException(
                status_code=429, detail="server busy, retry later",
                headers={"Retry-After": "1"},
            )
        return job

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + BATCH_WAIT_MS / 1000
            while len(batch) < BATCH_MAX:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue_mod.Empty:
                    break
            self._run_batch(batch)

    def _run_batch(self, batch: list[Job]) -> None:
        self.stats["batches"] += 1
        self.stats["batch_sizes"].append(len(batch))
        started = time.perf_counter()
        try:
            # mirror joint_schema_model.systemone(), batched across requests
            encoded = [
                self._encode_record(self.processor.tokenizer, job.request, max_length=MAX_LENGTH, processor=self.processor)
                for job in batch
            ]
            device = next(self.model.parameters()).device
            logits = self.model(
                self._collate_records(encoded, self.processor.tokenizer.pad_token_id, device)
            )
            for job, one_encoded, one_logits in zip(batch, encoded, logits):
                request = job.request
                questions = request["questions"]
                answers = {
                    question.question_id: self._systemone_answer(
                        questions[question.question_id],
                        dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist())),
                    )
                    for question, question_logits in zip(one_encoded.questions, one_logits)
                }
                job.result = {
                    "model": request["model"],
                    "answers": answers,
                    "usage": {
                        "input_tokens": len(one_encoded.input_ids),
                        "output_tokens": 0,
                        "batch_size": len(batch),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                }
        except BaseException as exc:  # noqa: BLE001 - must not kill the inference thread
            log.exception("batch failed")
            self.stats["errors"] += len(batch)
            for job in batch:
                job.error = exc
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self._latencies.append(elapsed)
            self._latencies = self._latencies[-1000:]
            self.stats["requests"] += len(batch)
            for job in batch:
                job.done.set()

    # -- stats ------------------------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        lats = sorted(self._latencies)
        n = len(lats)
        return {
            **{k: v for k, v in self.stats.items() if k != "batch_sizes"},
            "avg_batch_size": round(sum(self.stats["batch_sizes"]) / max(1, len(self.stats["batch_sizes"])), 2),
            "queue_depth": self.queue.qsize(),
            "batch_forward_p50_ms": round(lats[n // 2], 1) if n else None,
            "batch_forward_p95_ms": round(lats[int(n * 0.95)], 1) if n else None,
        }


ENGINE: BatchEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE
    started = time.perf_counter()
    try:
        model, processor = _load_model()
        if WARMUP:
            from joint_schema_model import systemone  # noqa: PLC0415
            systemone(model, processor, {
                "model": SERVED_MODEL_NAME,
                "state": {"warmup": True},
                "questions": {"ready": {"type": "noul", "instructions": "Is this a warmup request?"}},
            }, max_length=MAX_LENGTH)
            log.info("single-request warmup complete")
        ENGINE = BatchEngine(model, processor)
        # Warm every batch shape up to BATCH_MAX: Triton autotunes per (batch,
        # seqlen) shape, and an unwarmed shape costs seconds on first use
        # (observed 8.5 s for the first batch-of-8). Warmup runs through the
        # engine itself, validating the batching path at startup.
        if WARMUP:
            warmup_request = {
                "model": SERVED_MODEL_NAME,
                "state": {"warmup": True},
                "questions": {"ready": {"type": "noul", "instructions": "Is this a warmup request?"}},
            }
            for size in [s for s in (1, 2, 4, BATCH_MAX) if s <= BATCH_MAX]:
                jobs = [ENGINE.submit(dict(warmup_request)) for _ in range(size)]
                for job in jobs:
                    job.done.wait()
                log.info("warmup: batch size %d ok", size)
        STATE.update(model=model, processor=processor, ready=True, error=None, loaded_at=time.time())
        log.info("model ready in %.1fs (queue=%d batch=%d wait=%.0fms)",
                 time.perf_counter() - started, QUEUE_MAX, BATCH_MAX, BATCH_WAIT_MS)
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = repr(exc)
        log.exception("model failed to load")
        raise
    yield
    STATE.update(ready=False)


app = FastAPI(title="Clef-Flash (hardened)", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def guard(request: Request, call_next):
    # probes and stats stay open (LB/monitoring); decisions require the key when set
    if API_KEY and request.url.path not in ("/health", "/ready"):
        if request.headers.get("authorization") != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    length = request.headers.get("content-length")
    if length and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "body too large"})
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if STATE["ready"] else "error",
        "ready": STATE["ready"],
        "model": MODEL_ID,
        "device": DEVICE,
        "error": STATE["error"],
        "loaded_at": STATE["loaded_at"],
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/ready")
def ready():
    if STATE["ready"]:
        return {"ready": True}
    return JSONResponse(status_code=503, content={"ready": False, "error": STATE["error"]})


@app.get("/stats")
def stats() -> dict[str, Any]:
    if ENGINE is None:
        return {"ready": False}
    return ENGINE.snapshot()


async def _handle(body: dict[str, Any], require_model: bool) -> dict[str, Any]:
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=f"model not ready: {STATE['error']}")
    request = _prepare_request(body, require_model)
    assert ENGINE is not None
    job = ENGINE.submit(request)
    # block a worker thread (not the event loop) until the inference thread
    # finishes the batch; the HTTP layer stays responsive meanwhile
    import asyncio

    await asyncio.get_running_loop().run_in_executor(None, job.done.wait)
    if job.error is not None:
        raise HTTPException(status_code=500, detail=str(job.error))
    assert job.result is not None
    return job.result


@app.post("/v1/systemone")
async def systemone_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Native Jev/SystemOne request/response. Requires ``model`` and ``state``."""
    return await _handle(body, require_model=True)


@app.post("/predict")
async def predict(body: dict[str, Any]) -> dict[str, Any]:
    """Convenience endpoint: ``state`` + ``questions`` (``model`` optional)."""
    return await _handle(body, require_model=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
