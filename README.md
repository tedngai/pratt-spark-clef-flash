# Clef-Flash on NVIDIA GB10 (DGX Spark)

Dockerized inference for [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash)
on an NVIDIA GB10 / DGX Spark (aarch64, CUDA 13, 121 GB unified memory).

Clef-Flash is a 9B multimodal model that reads a `state` (text/JSON/images/video) plus a
schema of typed questions and returns **one probability per allowed option per question**
in a single forward pass. It is a decision/classification model, not a text generator.

## Why not `vllm serve` / SGLang

The HF page lists vLLM/SGLang tabs, but those are generic boilerplate. Clef-Flash adds a
custom **joint schema head** and its own record encoding in `joint_schema_model.py`, which
vLLM/SGLang do not know about. The supported path is the repository's own Python code,
which is what this project wraps in an HTTP service.

## Layout

```
clef-flash/
├── Dockerfile              # pinned base + transformers + fused kernels
├── compose.yaml            # canonical runtime definition
├── clef.env                # configuration (ports, paths, pins, build args)
├── server.py               # FastAPI service (SystemOne + /predict)
├── healthcheck.py          # strict readiness probe for compose/systemd
├── start_clef.sh           # compose up -d
├── stop_clef.sh            # compose down
├── prefetch_clef.sh        # download ~18 GB of weights into data/huggingface
├── install_systemd.sh      # install + enable the systemd unit
├── uninstall_systemd.sh    # remove the systemd unit
├── systemd/clef-flash.service  # unit template (paths substituted on install)
├── examples/               # curl + Python clients
├── tests/smoke.sh          # end-to-end check
└── data/huggingface/       # Hugging Face cache (created at runtime)
```

## Quick start

```bash
cd clef-flash
./prefetch_clef.sh      # builds the image (compiles kernels), downloads ~18 GB
./start_clef.sh
docker compose --env-file clef.env -f compose.yaml logs -f
```

Wait for `model ready in ...s` and `Qwen3.5 fused fast path available: True`, then:

```bash
./tests/smoke.sh
```

Everyday use:

```bash
./start_clef.sh         # compose up -d (no-op if already up)
./stop_clef.sh          # compose down
BUILD=1 ./start_clef.sh # force an image rebuild
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | status, device, load errors |
| `GET` | `/ready` | 200 only once the model is loaded (used by healthcheck) |
| `GET` | `/v1/models` | OpenAI-style model list |
| `POST` | `/v1/systemone` | native Jev/SystemOne body (requires `model` + `state`) |
| `POST` | `/predict` | same as above, `model` optional |

Example (`examples/systemone.sh`):

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
      },
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}
    }
  }' | python3 -m json.tool
```

Question types: `choice` (named options), `score` (ordered options), `noul` (true/false).
Images may be passed as `"images": ["https://...", "data:image/png;base64,..."]`; they are
decoded server-side into PIL images before encoding.

## Fused kernels (Qwen3.5 fast path)

The Qwen3.5 backbone mixes linear-attention (gated delta rule) and full-attention layers.
Without the fused kernels, transformers logs a warning and uses a pure-PyTorch fallback.
The image installs both libraries that `is_fast_path_available` requires:

| Package | What it provides | How installed |
| --- | --- | --- |
| `flash-linear-attention==0.5.2` | Triton kernels: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `FusedRMSNormGated` | pip wheel (pure Python + Triton) |
| `causal-conv1d==1.7.0` | compiled CUDA extension: `causal_conv1d_fn/update` | **compiled from source** at image build |

Because `causal-conv1d` is compiled at build time and the build has no GPU to auto-detect,
`CUDA_ARCH_LIST` in `clef.env` must match the target GPU:

| GPU | `CUDA_ARCH_LIST` |
| --- | --- |
| GB10 / DGX Spark | `12.1` |
| RTX 50 (Blackwell) | `12.0` |
| H100 / Hopper | `9.0` |
| A100 | `8.0` |

Set `INSTALL_KERNELS=0` to build without kernels (pure-PyTorch fallback). The image logs
`Qwen3.5 fused fast path available: True/False` at startup so you can confirm.

### Measured on this GB10

| | No kernels | Fused kernels |
| --- | --- | --- |
| Text decision, warm p50 | ~192 ms | **~109 ms** |
| First request after ready | 20 s+ (autotune) | **~128 ms** (startup warmup) |
| Image decision | ~900 ms | **~604 ms** |
| GPU / unified memory | ~18.5 GB | ~18.9 GB |
| Startup to ready | ~130 s | ~150 s (includes 20 s warmup) |

Decisions are unchanged (invoice example: `overdue` 0.973, `large` 0.974). `CLEF_WARMUP=1`
runs one synthetic decision at startup so Triton autotune happens before real traffic.


## Auto-start with systemd

```bash
sudo ./install_systemd.sh
systemctl status clef-flash
journalctl -u clef-flash -f
sudo ./uninstall_systemd.sh   # to remove
```

`install_systemd.sh` substitutes the real project path and docker binary into the unit
template, installs `/etc/systemd/system/clef-flash.service`, and enables it. The unit is
`Type=oneshot` + `RemainAfterExit=yes`: `systemctl start/stop/reload` map to
`docker compose up -d` / `down` / `restart`, and it starts on boot after `docker.service`.

## Deploy to another GB10 / DGX Spark

The project is self-contained and uses relative paths, so a copy runs anywhere.

1. Install Docker, the NVIDIA container toolkit, and the NVIDIA driver/CUDA on the target.
2. Copy the project (or `git clone`) to the target. Do **not** copy `data/` if you want a
   clean cache.
3. Edit `clef.env` if needed. For a GB10 keep `CUDA_ARCH_LIST=12.1`; change `PORT` if 8001
   is taken.
4. Build + download:
   ```bash
   ./prefetch_clef.sh
   ```
5. Start, optionally enable boot autostart:
   ```bash
   ./start_clef.sh
   sudo ./install_systemd.sh
   ```

The base image is pinned by multi-arch digest (`BASE_IMAGE` in `clef.env`), so the same
build works on x86_64 too — just set `CUDA_ARCH_LIST` for the target GPU and, if the torch
build differs, override `BASE_IMAGE` with an appropriate image.

## Configuration (`clef.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8001` | host port |
| `MODEL_ID` | `Cloudflare/clef-flash` | HF repo |
| `CLEF_MAX_LENGTH` | `16384` | token budget for state + schema |
| `CLEF_WARMUP` | `1` | run one synthetic decision at startup (compiles kernels) |
| `TRANSFORMERS_VERSION` | `5.10.2` | pinned transformers |
| `INSTALL_KERNELS` | `1` | build fused kernels |
| `CUDA_ARCH_LIST` | `12.1` | target GPU arch for the compile |
| `RESTART_POLICY` | `unless-stopped` | compose restart policy |
| `HF_HOME_HOST` | `./data/huggingface` | weights cache (relative, portable) |
| `HF_TOKEN` | unset | only for gated/private models |
| `HF_HUB_DISABLE_XET` | unset | set to `1` if Xet downloads stall |

## Troubleshooting

- **`/health` shows `ready: false`** — usually a partial weight download. Check
  `du -sh data/huggingface` (should be ~18 GB) and re-run `./prefetch_clef.sh`.
- **Kernel compile fails** — confirm `CUDA_ARCH_LIST` matches the GPU and the base image
  has `nvcc` (it does). As a fallback set `INSTALL_KERNELS=0` and rebuild.
- **Startup is slow** — first load reads ~18 GB from disk and warms up (~2 min). Later
  starts are faster.
- **Out of memory** — stop other GPU workloads (video models) before starting.
- **Port already in use** — change `PORT` in `clef.env`.

## Notes

- The base image intentionally reuses the pinned `vllm/vllm-openai` arm64 image because it
  already provides torch 2.11.0+cu130 — the exact torch version the model card was tested
  with — plus `nvcc`/triton for the kernel build. vLLM itself is not used.
- `transformers` and the kernels live in an isolated venv (`/opt/clef-venv`,
  `--system-site-packages`) so the base image's vLLM environment is untouched; torch is
  inherited from the base.


## Hardened server (micro-batching + bounded queue)

`server_hardened.py` is a drop-in replacement for `server.py` that adds:

1. **Dedicated inference thread** — the GPU is owned by one thread, eliminating
   the intermittent HTTP 500s the stock server produces under concurrent load
   (concurrent callers into the shared model are a data race).
2. **Dynamic micro-batching** — requests arriving within `BATCH_WAIT_MS` are
   collated (via `collate_records`) into a single forward pass. A 9B bf16
   backbone costs ~66 ms/forward just in weight reads, so batching amortizes
   the dominant cost. Measured on the GB10 (3-option schema, BATCH_MAX=8):
   **7.1 -> 13.0 req/s (~1.8x) aggregate throughput**; the ceiling is
   per-record compute, so gains grow for shorter states and larger
   `BATCH_MAX`.
3. **Bounded queue** — beyond `QUEUE_MAX` waiting requests the server sheds
   load with `429` + `Retry-After` instead of letting latency grow unbounded
   (measured: a 100-request burst admits 72, sheds 28 with `429`, no 500s).
4. **`/stats`** — queue depth, average batch size, forward-pass p50/p95.
5. **Guards** — optional `CLEF_API_KEY` auth on POST endpoints, 13 MiB body
   cap, image count/pixel caps.
6. **Batch-safety limits** — requests with media, or estimated above
   `SOLO_TOKEN_LIMIT` (~512 tokens), always run solo; gathered batches are
   capped at `BATCH_TOKEN_LIMIT` (2048) total tokens. Rationale: a
   batch-8 forward at ~1k-token states triggered a multi-minute Triton
   autotune on GB10 whose GIL-bound compile threads starved uvicorn and
   sshd (full machine wedge, observed 2026-10-02). Long states are
   compute-bound anyway, so solo execution costs little throughput.
7. **Known gap** — Docker does not restart unhealthy containers. If the
   GPU wedges, the compose healthcheck marks it unhealthy but the
   container stays up; add an autoheal sidecar or a systemd timer that
   restarts on `unhealthy` for unattended production.

Configuration lives in `clef.env` (`CLEF_SERVER_FILE`, `QUEUE_MAX`,
`BATCH_MAX`, `BATCH_WAIT_MS`, `CLEF_API_KEY`). Set
`CLEF_SERVER_FILE=server` and restart to revert to the stock server.
`BATCH_WAIT_MS` is a pure latency tax when idle (10 ms default); lower it for
latency-critical deployments or raise `BATCH_MAX` for throughput-critical ones.
