"""Batching experiments on Decision-2.0-Lux-9B (runs inside the lux9b container).

E1  Multi-record batch amortization: 8 records through system_one one-by-one
    vs the same encoded rows re-collated into ONE forward. Rows are captured
    by spying `decision_model.collate` (the module the serving path lazily
    imports from on every request) — no signature guessing.
E2  share_context=False vs True on a multi-question request: latency + answers.
E3  Batch invariance: same question answered solo vs inside an 8-row collate
    (and via share_context) — choice flips + max probability delta.

Prints JSON to stdout; also saved next to this file.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "vllm-sr/Decision-2.0-Lux-9B"
RESULTS: dict = {"errors": []}


def load_model():
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cuda")


def sync():
    torch.cuda.synchronize()


def bench(fn, reps: int = 20) -> dict:
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(statistics.median(times), 1), "mean": round(statistics.mean(times), 1)}


def main() -> None:
    t0 = time.perf_counter()
    model = load_model()
    print(f"model loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    # HF wrapper -> runtime (api.Decision2, share_context-aware) -> qwen backend
    rt = getattr(model, "runtime", model)
    backend = getattr(rt, "backend", rt)
    bmodel = getattr(backend, "model", None)
    device = getattr(backend, "device", torch.device("cuda"))
    if bmodel is None:
        RESULTS["errors"].append("no backbone reachable via runtime.backend.model")
        print(json.dumps(RESULTS, indent=2))
        return

    # ---- locate the vendored decision_model module and spy its collate ----
    dm = next(
        (m for n, m in sys.modules.items()
         if m is not None and n.endswith("dev2model.decision_model")),
        None,
    )
    if dm is None or not hasattr(dm, "collate"):
        RESULTS["errors"].append(
            "dev2model.decision_model not found in sys.modules; "
            f"candidates: {[n for n in sys.modules if 'dev2model' in n][:10]}"
        )
        print(json.dumps(RESULTS, indent=2))
        return
    orig_collate = dm.collate

    sys.path.insert(0, str(Path(__file__).parent))
    from data_bank import load_banking_rows, small_criteria  # noqa: PLC0415

    captured: list = []
    captured_pad: list = []

    def spy(encs, pad, *a, **kw):
        if isinstance(encs, list) and len(encs) == 1:
            captured.append(dict(encs[0]))
            captured_pad.append(pad)
        return orig_collate(encs, pad, *a, **kw)

    dm.collate = spy

    def capture_row(state, questions) -> dict:
        captured.clear()
        captured_pad.clear()
        rt.system_one(state=state, questions=questions)
        if not captured:
            raise RuntimeError("collate spy did not fire for single-question request")
        return captured[0]

    def ask(state, questions, sc=None):
        """system_one -> answers dict (handles both dict and tuple return shapes)."""
        out = (
            rt.system_one(state=state, questions=questions, share_context=sc)
            if sc is not None
            else rt.system_one(state=state, questions=questions)
        )
        return out["answers"] if isinstance(out, dict) else out[0]

    def batched_forward_ms(encs_list, pad) -> float:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in orig_collate(encs_list, pad).items()
        }
        sync()
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bmodel(**batch)
        sync()
        return (time.perf_counter() - t) * 1000

    # ================= E1: multi-record amortization =================
    for label, n_opts in [("small_3opt", 3), ("banking77_77opt", 77)]:
        try:
            if n_opts == 3:
                states, questions = small_criteria()
            else:
                rows = load_banking_rows(8)
                states = [r["state"] for r in rows]
                questions = {"q": rows[0]["questions"]["q"]}

            encs = [capture_row(s, questions) for s in states]
            pad = captured_pad[-1]

            rt.system_one(state=states[0], questions=questions)  # warmup
            batched_forward_ms(encs[:2], pad)  # warmup

            serial = bench(
                lambda: [rt.system_one(state=s, questions=questions) for s in states],
                reps=6,
            )
            t_enc = time.perf_counter()
            for s in states:
                capture_row(s, questions)
            capture_8_ms = (time.perf_counter() - t_enc) * 1000
            batched = bench(lambda: batched_forward_ms(encs, pad), reps=6)

            RESULTS[f"E1_{label}"] = {
                "records": 8,
                "serial_total_p50_ms": serial["p50"],
                "serial_per_record_ms": round(serial["p50"] / 8, 1),
                "batched_collate_forward_p50_ms": batched["p50"],
                "eight_full_serial_calls_ms": round(capture_8_ms, 1),
                "amortization_serial_vs_batched_forward": round(serial["p50"] / batched["p50"], 2),
            }
            print(f"E1 {label}: {RESULTS[f'E1_{label}']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            RESULTS["errors"].append(f"E1_{label}: {exc!r}")
            print(f"E1 {label} FAILED: {exc!r}", flush=True)

    # ================= E2: share_context =================
    try:
        state = (
            "The monitoring dashboard flagged elevated error rates on the checkout service "
            "and payments began failing for about twelve minutes. The team rolled back a "
            "deployment and error rates returned to baseline. "
        ) * 4
        questions = {
            f"q{i}": {
                "type": "choice",
                "instructions": f"Does the message mention issue {i}?",
                "criteria": {"yes": "It is explicitly mentioned", "no": "It is not mentioned"},
            }
            for i in range(8)
        }
        rt.system_one(state=state, questions=questions, share_context=False)  # warmup
        off = bench(lambda: rt.system_one(state=state, questions=questions, share_context=False))
        ans_off = ask(state, questions, False)
        try:
            rt.system_one(state=state, questions=questions, share_context=True)  # warmup
            on = bench(lambda: rt.system_one(state=state, questions=questions, share_context=True))
            ans_on = ask(state, questions, True)
            flips = sum(1 for k in ans_off if ans_off[k]["choice"] != ans_on.get(k, {}).get("choice"))
            deltas = [
                abs(ans_off[k]["probabilities"][pk] - ans_on[k]["probabilities"][pk])
                for k in ans_off
                if k in ans_on
                for pk in ans_off[k]["probabilities"]
                if pk in ans_on[k].get("probabilities", {})
            ]
            RESULTS["E2_share_context"] = {
                "questions": 8,
                "off_p50_ms": off["p50"],
                "on_p50_ms": on["p50"],
                "speedup": round(off["p50"] / on["p50"], 2),
                "decision_flips": flips,
                "max_prob_delta": round(max(deltas), 4) if deltas else 0,
            }
        except Exception as exc:  # noqa: BLE001
            RESULTS["E2_share_context"] = {"off_p50_ms": off["p50"], "share_context_error": repr(exc)[:200]}
        print(f"E2: {RESULTS.get('E2_share_context')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS["errors"].append(f"E2: {exc!r}")
        print(f"E2 FAILED: {exc!r}", flush=True)

    # ================= E3: batch invariance =================
    try:
        rows = load_banking_rows(12)
        states = [r["state"] for r in rows]
        q77 = rows[0]["questions"]["q"]
        fillers = {
            f"f{j}": {
                "type": "choice",
                "instructions": f"Filler check {j}: does the text mention refunds?",
                "criteria": {"yes": "It is explicitly mentioned", "no": "It is not mentioned"},
            }
            for j in range(7)
        }

        solo = [ask(s, {"q": q77})["q"] for s in states]

        flips, max_delta = 0, 0.0
        for i, s in enumerate(states):
            mixed = dict(fillers)
            mixed["q"] = q77
            a_mixed = ask(s, mixed)["q"]
            if a_mixed["choice"] != solo[i]["choice"]:
                flips += 1
            d = max(
                abs(a_mixed["probabilities"][pk] - solo[i]["probabilities"][pk])
                for pk in solo[i]["probabilities"]
                if pk in a_mixed["probabilities"]
            )
            max_delta = max(max_delta, d)

        sc_flips = 0
        try:
            for i, s in enumerate(states):
                a_sc = ask(s, {"q": q77}, True)["q"]
                if a_sc["choice"] != solo[i]["choice"]:
                    sc_flips += 1
            sc_note = "ok"
        except Exception as exc:  # noqa: BLE001
            sc_note = repr(exc)[:120]

        RESULTS["E3_invariance"] = {
            "probes": len(states),
            "flips_solo_vs_8row_batch": flips,
            "max_prob_delta_batch": round(max_delta, 4),
            "flips_solo_vs_share_context": sc_flips if sc_note == "ok" else None,
            "share_context_note": sc_note,
        }
        print(f"E3: {RESULTS['E3_invariance']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS["errors"].append(f"E3: {exc!r}")
        print(f"E3 FAILED: {exc!r}", flush=True)

    dm.collate = orig_collate  # restore

    out = Path(__file__).parent / "batch_experiment.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
