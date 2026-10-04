# Batching Comparison: clef-flash vs Decision-2.0-Lux-9B

*Follow-up to the three-way benchmark (`report.md`). All Lux experiments ran on the GB10
(192.168.1.242) against the vendor runtime inside the `lux9b` container, using the exact
serving path (`system_one` → per-question rows → `micro_batches` → `collate` → bf16 autocast
forward). clef-flash numbers come from the hardened server's stress/latency profiles
(`results/stress_hardened.json`, `results/latency.json`).*

## Headline

| Capability | clef-flash (hardened server) | Decision-2.0-Lux-9B (vendor runtime) |
|---|---|---|
| Cross-request batching (multi-record) | **Yes — automatic** at the HTTP layer: 7.1 → 13.0 req/s at batch 8 on light schemas (**1.8×**) | **Not exposed.** `system_one` serves one request at a time; their own eval harness states "one request at a time". Internal `collate` could amortize **2.36×** (light) / **1.05×** (heavy) if a server exposed it |
| Within-request batching (multi-question) | Native (per-question rows in one collate) | Native (`micro_batches` token budget) |
| Shared-prefix prefill (`share_context`) | n/a | Opt-in, **off by default**; **2.42×–5.91×** for 8-question requests when engaged; **silently disabled on transformers ≠ 5.17.x** |
| Batch invariance | Exact path | **Not batch-invariant**: 1/12 probes flipped choice when co-questions changed (max Δp 1.31pp) |
| Batch-safety | Solo-token guard (512 tok) + batch cap (2048 tok) after the Triton-autotune wedge | `forward_token_budget` element cap (2³⁰) splits oversized batches; ROCm-only fast path inert on GB10 |

## E1 — Multi-record batch amortization (Lux, vendor collate, bf16)

8 single-question records timed serially (8 × `system_one`) vs the same encoded rows
re-collated into **one** forward. Rows captured by spying `decision_model.collate` during
real requests — the exact tensors the serving path produces.

| Schema | Serial (8 calls) | Per record | Batched (1 forward) | Amortization |
|---|---|---|---|---|
| Small (3-option, ~60 tok rows) | 821.8 ms | 102.7 ms | 348.3 ms | **2.36×** |
| BANKING77 (77-option, ~600 tok rows) | 4791.2 ms | 598.9 ms | 4583.5 ms | **1.05×** |

**Interpretation.** Both models share the same architectural property: every question is its
own row, so the state+instructions prompt is re-run through the backbone once per row. On
short rows, batching mainly amortizes kernel-launch/fixed overheads (~2.4×); on heavy rows
the per-row backbone compute dominates and batching buys almost nothing (1.05×). clef-flash's
1.8× HTTP-level gain at batch 8 (light schema) is the same physics; its server just exposes it
to clients automatically. Neither model escapes the per-row prompt-recompute cost.

## E2 — `share_context` (shared-prefix prefill)

8 yes/no questions over one shared state. Three configurations: exact path,
`share_context=True` (tree policy), explicit `{"mode": "cache"}`.

### Deployed container (transformers 5.18.0): sharing never engages

`backend.share_stats.reason = "transformers 5.18.0 is not verified"` — the runtime hard-gates
the feature to `transformers 5.17.x` (`TESTED_TRANSFORMERS = ("5.17.",)`). Requests fall back
to the exact path: **bit-identical answers, 1.00× latency**, at any state length. The failure
is silent unless you read `share_stats`.

### Pinned transformers 5.17.0: sharing engages

| State (8 questions) | Exact | Tree policy | Cache policy | Flips | Max Δp |
|---|---|---|---|---|---|
| Short (117-tok shared prefix) | 487.0 ms | **201.2 ms (2.42×)** | 286.4 ms (1.70×) | 0 | 0.0031 |
| Long (1678-tok shared prefix) | 4423.2 ms | **748.6 ms (5.91×)** | 793.7 ms (5.57×) | 0 | 0.0020 |

- Engagement confirmed via `share_stats` (`shared: true`, `prefix_tokens` 117 / 1678).
- Auto break-even on this backbone (`qwen3_5_text`, no HIP graphs on GB10): **512 tokens
  saved** — (questions − 1) × prefix must exceed it, or the request silently runs exact.
- Answer drift vs exact path is small (≤ 0.31pp) and produced **no decision flips** on these
  probes — smaller than the exact path's own batch-composition drift (E3).
- Docstring caveat stands: shared-path arithmetic differs from exact (other GEMM shapes /
  kernels / reduction order), so near-ties *can* flip; `tau` policy can re-score low-margin
  answers on the exact path.

**Operational note:** to get this in production the lux container must pin
`transformers==5.17.*` (our `start_lux.sh` installs `>=5.17` → 5.18.0 → feature inert).

## E3 — Batch invariance (exact path)

12 real BANKING77 probes, each answered (a) solo, (b) inside an 8-row request with 7 filler
questions (different collate composition/padding).

| Probe | Result |
|---|---|
| Choice flips solo vs 8-row batch | **1 / 12** |
| Max probability delta | **0.0131** (1.31pp) |
| Flips solo vs `share_context=True` | 0 / 12 (single question < `min_questions=2` → exact path) |

The vendor's own caveat is confirmed empirically: the exact path itself is **not
batch-invariant** — co-questions in the same request can flip a decision. Anyone pinning
decisions for audit should pin the full request composition, not just the question.

## Server-level contrast (HTTP)

- **clef-flash**: `collate_records` micro-batches concurrent requests with token budgets and
  solo-routing for long states; measured 1.8× throughput at concurrency 8 (light schema).
- **Lux-9B**: our SystemOne-compatible wrapper (`lux_server.py`) serializes requests with a
  lock — no cross-request batching; the vendor API offers no batch endpoint. The 2.36×
  amortization measured in E1 is *reachable but unexposed*.

## Bottom line

Same core limitation, different serving story. Both models pay per-question prompt recompute,
so batching only pays on light schemas (~2×) and collapses on heavy ones (~1.05×). clef-flash
captures the available win automatically at the server layer; Lux leaves it on the table at
the HTTP layer but offers a bigger, opt-in intra-request win (`share_context`, up to ~6× on
long shared states) that is fragile in practice — silently disabled unless transformers is
pinned to the one verified minor version.
