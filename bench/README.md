# Clef-flash local benchmark vs Jev / decision models

Benchmarks the locally-hosted **Cloudflare/clef-flash** decision model
(`http://192.168.1.242:8001`, NVIDIA GB10 / DGX Spark, via the
[pratt-spark-clef-flash](https://github.com/tedngai/pratt-spark-clef-flash) server)
against **Jev** and other decision models from the published **Jev Decision Index**.

## Datasets (all public test sets)

| Dataset | Task | n | Metric | Decision Index row |
| --- | --- | ---: | --- | --- |
| BANKING77 | 77-way intent | 3,080 | macro-F1 | BANKING77 |
| CLINC150+OOS | 151-way intent incl. out-of-scope | 5,500 | macro-F1 | CLINC150+OOS |
| ANLI (R1+R2+R3) | 3-way NLI | 3,200 | macro-F1 | ANLI (clef card) |

Total: 11,780 decisions through `POST /v1/systemone`, one question per request.

## Files

- `fetch_data.py` — downloads the three test sets to `data/*.jsonl`
- `run_bench.py` — sends every sample to the endpoint; appends to `results/<dataset>.jsonl`; **resumable** (re-run skips completed ids)
- `latency_bench.py` — microbenchmark: state-size scaling, question-count scaling, image, concurrency (writes `results/latency.json`)
- `analyze.py` — metrics + charts + `report/report.md`
- `decision_index.py` — published reference numbers (Cloudflare blog + model cards)

## Reproduce

```
python fetch_data.py
python run_bench.py --dataset anli --concurrency 1
python run_bench.py --dataset banking77 --concurrency 1
python run_bench.py --dataset clinc150 --concurrency 1   # ~75 min
python latency_bench.py                                  # run when GPU is idle
python analyze.py
```

## Findings baked into the harness

- The server processes requests **serially** on the GPU: concurrency 4 gives ~4x
  per-request latency with no throughput gain. Run benchmarks with `--concurrency 1`.
- Concurrent large-schema requests occasionally returned HTTP 500 (thread-safety of
  the model call); serial runs had zero failures.
- Latency scales strongly with option count (joint schema head):
  ~110 ms (3 options) → ~500 ms (77 options) → ~840 ms (151 options).
- The server caps state+schema at `CLEF_MAX_LENGTH=16384` tokens.

## Caveats

- Published Decision Index numbers come from Cloudflare/TypeSafe's own harness and
  prompt encodings. Our macro-F1 for clef-flash is exact for *this* endpoint and
  *our* encodings; cross-harness deltas vs published rows are approximate.
- A live Jev head-to-head under identical encodings would need a TypeSafe API key
  (`POST https://api.typesafe.ai/v1/systemone`); `run_bench.py --base-url` can be
  pointed at any SystemOne-compatible host, so the same harness works unchanged.
