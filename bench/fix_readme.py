#!/usr/bin/env python3
"""Sync README.md on the server with measured stress-test numbers."""
from pathlib import Path

p = Path("README.md")
text = p.read_text()

old_gain = """   backbone costs ~66 ms/forward just in weight reads, so batching amortizes
   the dominant cost: measured 4-8x aggregate throughput for short schemas."""
new_gain = """   backbone costs ~66 ms/forward just in weight reads, so batching amortizes
   the dominant cost. Measured on the GB10 (3-option schema, BATCH_MAX=8):
   **7.1 -> 13.0 req/s (~1.8x) aggregate throughput**; the ceiling is
   per-record compute, so gains grow for shorter states and larger
   `BATCH_MAX`."""
assert text.count(old_gain) == 1, "gain paragraph anchor not found"
text = text.replace(old_gain, new_gain)

old_queue = """3. **Bounded queue** — beyond `QUEUE_MAX` waiting requests the server sheds
   load with `429` + `Retry-After` instead of letting latency grow unbounded."""
new_queue = """3. **Bounded queue** — beyond `QUEUE_MAX` waiting requests the server sheds
   load with `429` + `Retry-After` instead of letting latency grow unbounded
   (measured: a 100-request burst admits 72, sheds 28 with `429`, no 500s)."""
assert text.count(old_queue) == 1, "queue paragraph anchor not found"
text = text.replace(old_queue, new_queue)

p.write_text(text)
print("README.md: synced measured numbers")
