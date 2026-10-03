#!/usr/bin/env python3
"""Follow-up patch: batch-safety env knobs + README guards (after the wedge incident)."""
from pathlib import Path


def patch(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if new in text:
        print(f"{path}: already patched")
        return
    assert text.count(old) == 1, f"{path}: anchor not unique/found: {old[:60]!r}"
    p.write_text(text.replace(old, new))
    print(f"{path}: patched")


# 1. compose.yaml — pass the new knobs into the container
patch(
    "compose.yaml",
    "      BATCH_WAIT_MS: ${BATCH_WAIT_MS:-10}\n",
    "      BATCH_WAIT_MS: ${BATCH_WAIT_MS:-10}\n"
    "      SOLO_TOKEN_LIMIT: ${SOLO_TOKEN_LIMIT:-512}\n"
    "      BATCH_TOKEN_LIMIT: ${BATCH_TOKEN_LIMIT:-2048}\n",
)

# 2. clef.env — document + set them
env = Path("clef.env")
if "SOLO_TOKEN_LIMIT" in env.read_text():
    print("clef.env: already patched")
else:
    block = """# Batch-safety: requests estimated above SOLO_TOKEN_LIMIT tokens (or with
# images/videos) run solo; a gathered batch is capped at BATCH_TOKEN_LIMIT
# total. Avoids the multi-minute Triton autotune that large-batch x
# long-sequence shapes trigger on GB10 (observed to starve the whole box).
SOLO_TOKEN_LIMIT=512
BATCH_TOKEN_LIMIT=2048
"""
    with env.open("a") as fh:
        fh.write(block)
    print("clef.env: appended batch-safety block")

# 3. README — safety guards bullet + known gap
patch(
    "README.md",
    "5. **Guards** — optional `CLEF_API_KEY` auth on POST endpoints, 13 MiB body\n   cap, image count/pixel caps.",
    "5. **Guards** — optional `CLEF_API_KEY` auth on POST endpoints, 13 MiB body\n"
    "   cap, image count/pixel caps.\n"
    "6. **Batch-safety limits** — requests with media, or estimated above\n"
    "   `SOLO_TOKEN_LIMIT` (~512 tokens), always run solo; gathered batches are\n"
    "   capped at `BATCH_TOKEN_LIMIT` (2048) total tokens. Rationale: a\n"
    "   batch-8 forward at ~1k-token states triggered a multi-minute Triton\n"
    "   autotune on GB10 whose GIL-bound compile threads starved uvicorn and\n"
    "   sshd (full machine wedge, observed 2026-10-02). Long states are\n"
    "   compute-bound anyway, so solo execution costs little throughput.\n"
    "7. **Known gap** — Docker does not restart unhealthy containers. If the\n"
    "   GPU wedges, the compose healthcheck marks it unhealthy but the\n"
    "   container stays up; add an autoheal sidecar or a systemd timer that\n"
    "   restarts on `unhealthy` for unattended production.",
)
