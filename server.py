"""Clef-Flash HTTP API.

Serves Cloudflare/clef-flash (a Qwen3.5-9B backbone plus a joint schema head) through
the repository's native SystemOne interface, plus a small convenience ``/predict``
endpoint.

The model is not a text generator: each request returns one probability per allowed
option for every typed question. ``/v1/systemone`` mirrors the Jev/SystemOne request
and response bodies; ``/predict`` is the same thing without the required ``model``
field.

Images may be supplied as http(s) URLs, base64 strings, or ``data:`` URIs.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
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

logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("clef.server")

STATE: dict[str, Any] = {
    "model": None,
    "processor": None,
    "systemone": None,
    "ready": False,
    "error": None,
    "loaded_at": None,
}


def _load_model() -> tuple[Any, Any, Any]:
    """Download (or reuse cached) weights and load the backbone + joint head."""
    log.info("resolving %s from the Hugging Face cache", MODEL_ID)
    path = snapshot_download(MODEL_ID)
    if path not in sys.path:
        sys.path.insert(0, path)
    # Imported from the model repository itself; see Cloudflare/clef-flash.
    from joint_schema_model import load_release_model, systemone  # noqa: PLC0415

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: PLC0415

        log.info(
            "Qwen3.5 fused fast path available: %s",
            modeling_qwen3_5.is_fast_path_available,
        )
    except Exception:  # noqa: BLE001
        log.debug("could not determine Qwen3.5 fast-path status", exc_info=True)

    log.info("loading Clef-Flash on %s", DEVICE)
    model, processor = load_release_model(path, device=DEVICE)
    return model, processor, systemone


def _warmup(model: Any, processor: Any, systemone: Any) -> None:
    """Run one synthetic decision so Triton/CUDA kernels compile before real traffic."""
    if not WARMUP:
        log.info("warmup disabled (CLEF_WARMUP=0)")
        return
    request = {
        "model": SERVED_MODEL_NAME,
        "state": {"warmup": True},
        "questions": {"ready": {"type": "noul", "instructions": "Is this a warmup request?"}},
    }
    started = time.perf_counter()
    systemone(model, processor, request, max_length=MAX_LENGTH)
    log.info("warmup complete in %.1fs (fused kernels compiled)", time.perf_counter() - started)


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
        return Image.open(io.BytesIO(raw)).convert("RGB")
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
        request["images"] = [_decode_image(image) for image in images]
    return request


def _run(body: dict[str, Any], require_model: bool) -> dict[str, Any]:
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=f"model not ready: {STATE['error']}")
    request = _prepare_request(body, require_model)
    started = time.perf_counter()
    result = STATE["systemone"](STATE["model"], STATE["processor"], request, max_length=MAX_LENGTH)
    result.setdefault("usage", {})["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    try:
        model, processor, systemone = _load_model()
        _warmup(model, processor, systemone)
        STATE.update(
            model=model,
            processor=processor,
            systemone=systemone,
            ready=True,
            error=None,
            loaded_at=time.time(),
        )
        log.info("model ready in %.1fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = repr(exc)
        log.exception("model failed to load")
        raise
    yield
    STATE.update(model=None, processor=None, systemone=None, ready=False)


app = FastAPI(title="Clef-Flash", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Strict readiness probe: 200 only once the model is loaded."""
    if STATE["ready"]:
        return {"ready": True}
    return JSONResponse(status_code=503, content={"ready": False, "error": STATE["error"]})


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "cloudflare"}],
    }


@app.post("/v1/systemone")
def systemone_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Native Jev/SystemOne request/response. Requires ``model`` and ``state``."""
    return _run(body, require_model=True)


@app.post("/predict")
def predict(body: dict[str, Any]) -> dict[str, Any]:
    """Convenience endpoint: ``state`` + ``questions`` (``model`` optional)."""
    return _run(body, require_model=False)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
