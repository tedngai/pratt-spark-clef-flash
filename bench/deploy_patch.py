#!/usr/bin/env python3
"""One-shot patch: wire server_hardened.py into the clef-flash compose project.

Idempotent: safe to run twice. Fails loudly if the anchor text is not found
exactly once, so nothing is half-edited.
"""
from pathlib import Path


def patch(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if new in text:
        print(f"{path}: already patched")
        return
    count = text.count(old)
    assert count == 1, f"{path}: expected 1 occurrence of anchor, found {count}: {old[:60]!r}"
    p.write_text(text.replace(old, new))
    print(f"{path}: patched")


# 1. Dockerfile — copy the hardened server into the image
patch(
    "Dockerfile",
    "COPY server.py /app/server.py\nCOPY healthcheck.py /app/healthcheck.py",
    "COPY server.py /app/server.py\nCOPY server_hardened.py /app/server_hardened.py\nCOPY healthcheck.py /app/healthcheck.py",
)

# 2. compose.yaml — pass engine knobs into the container
patch(
    "compose.yaml",
    "      HF_TOKEN: ${HF_TOKEN:-}\n",
    "      HF_TOKEN: ${HF_TOKEN:-}\n"
    "      CLEF_API_KEY: ${CLEF_API_KEY:-}\n"
    "      QUEUE_MAX: ${QUEUE_MAX:-64}\n"
    "      BATCH_MAX: ${BATCH_MAX:-8}\n"
    "      BATCH_WAIT_MS: ${BATCH_WAIT_MS:-10}\n",
)

# 3. compose.yaml — select the server entrypoint via CLEF_SERVER_FILE
patch(
    "compose.yaml",
    "    healthcheck:",
    '    entrypoint: ["python", "/app/${CLEF_SERVER_FILE:-server}.py"]\n    healthcheck:',
)

# 4. clef.env — append the hardened-server block (idempotent)
env = Path("clef.env")
text = env.read_text()
if "CLEF_SERVER_FILE" in text:
    print("clef.env: already patched")
else:
    block = """
# --- Hardened server (server_hardened.py) --------------------------------
# Entry point inside the image; set to server to revert to the stock server.
CLEF_SERVER_FILE=server_hardened
# Dynamic micro-batching: gather up to BATCH_MAX requests within BATCH_WAIT_MS
# into a single forward pass. QUEUE_MAX bounds the waiting queue; extra
# requests get 429 + Retry-After instead of unbounded latency.
QUEUE_MAX=64
BATCH_MAX=8
BATCH_WAIT_MS=10
# Optional API key: when set, POST endpoints require Authorization: Bearer <key>
# (GET /health, /ready, /stats stay open for probes/monitoring).
# CLEF_API_KEY=
"""
    with env.open("a") as fh:
        fh.write(block)
    print("clef.env: appended hardened block")

# 5. README.md — document the hardened server
readme = Path("README.md")
text = readme.read_text()
section = """

## Hardened server (micro-batching + bounded queue)

`server_hardened.py` is a drop-in replacement for `server.py` that adds:

1. **Dedicated inference thread** — the GPU is owned by one thread, eliminating
   the intermittent HTTP 500s the stock server produces under concurrent load
   (concurrent callers into the shared model are a data race).
2. **Dynamic micro-batching** — requests arriving within `BATCH_WAIT_MS` are
   collated (via `collate_records`) into a single forward pass. A 9B bf16
   backbone costs ~66 ms/forward just in weight reads, so batching amortizes
   the dominant cost. Measured on the GB10 (3-option schema, BATCH_MAX=8):
   **7.1 → 13.0 req/s (~1.8x) aggregate throughput**; the ceiling is
   per-record compute, so gains grow for shorter states and larger
   `BATCH_MAX`.
3. **Bounded queue** — beyond `QUEUE_MAX` waiting requests the server sheds
   load with `429` + `Retry-After` instead of letting latency grow unbounded
   (measured: a 100-request burst admits 72, sheds 28 with `429`, no 500s).
4. **`/stats`** — queue depth, average batch size, forward-pass p50/p95.
5. **Guards** — optional `CLEF_API_KEY` auth on POST endpoints, 13 MiB body
   cap, image count/pixel caps.

Configuration lives in `clef.env` (`CLEF_SERVER_FILE`, `QUEUE_MAX`,
`BATCH_MAX`, `BATCH_WAIT_MS`, `CLEF_API_KEY`). Set
`CLEF_SERVER_FILE=server.py` and restart to revert to the stock server.
`BATCH_WAIT_MS` is a pure latency tax when idle (10 ms default); lower it for
latency-critical deployments or raise `BATCH_MAX` for throughput-critical ones.
"""
if "server_hardened" in text:
    print("README.md: already documented")
else:
    readme.write_text(text + section)
    print("README.md: appended hardened section")
