"""E2c: share_context with transformers 5.17.x pinned (the verified release).

Two states x {exact, tree, cache}: latency, decision flips, prob deltas,
plus backend.share_stats after each config to confirm engagement.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch


def sync():
    torch.cuda.synchronize()


def bench(fn, reps: int) -> dict:
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(statistics.median(ts), 1), "mean": round(statistics.mean(ts), 1)}


def main() -> None:
    import transformers
    from transformers import AutoModel

    model = AutoModel.from_pretrained("vllm-sr/Decision-2.0-Lux-9B", trust_remote_code=True, device_map="cuda")
    rt = model.runtime
    backend = rt.backend

    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Does the message mention issue {i}?",
            "criteria": {"yes": "It is explicitly mentioned", "no": "It is not mentioned"},
        }
        for i in range(8)
    }
    short_state = (
        "The monitoring dashboard flagged elevated error rates on the checkout service "
        "and payments began failing for about twelve minutes. The team rolled back a "
        "deployment and error rates returned to baseline. "
    ) * 3
    long_state = (
        "The monitoring dashboard flagged elevated error rates on the checkout service and "
        "payments began failing for about twelve minutes. The on-call engineer paged the "
        "payments team, who rolled back the last deployment. Error rates returned to baseline "
        "within five minutes. A postmortem was scheduled and the incident was closed the same "
        "evening. The follow-up actions included adding a canary stage and an automatic rollback "
        "trigger tied to the payment error budget. "
    ) * 20

    res: dict = {"transformers_version": transformers.__version__}

    for label, state, reps in (("short_130tok", short_state, 12), ("long_2200tok", long_state, 8)):
        def ask(sc):
            out = rt.system_one(state=state, questions=questions, share_context=sc)
            return out["answers"] if isinstance(out, dict) else out[0]

        for sc in (False, True, {"mode": "cache"}):  # warmup/compile
            ask(sc)

        cfg: dict = {}
        cfg["exact"] = {"time": bench(lambda: ask(False), reps=reps)}
        ans_exact = ask(False)
        for name, sc in (("tree", True), ("cache", {"mode": "cache"})):
            cfg[name] = {"time": bench(lambda: ask(sc), reps=reps)}
            ans = ask(sc)
            flips = sum(1 for k in ans_exact if ans_exact[k]["choice"] != ans.get(k, {}).get("choice"))
            deltas = [
                abs(ans_exact[k]["probabilities"][pk] - ans[k]["probabilities"][pk])
                for k in ans_exact
                if k in ans
                for pk in ans_exact[k]["probabilities"]
                if pk in ans[k].get("probabilities", {})
            ]
            cfg[name]["speedup_vs_exact"] = round(cfg["exact"]["time"]["p50"] / cfg[name]["time"]["p50"], 2)
            cfg[name]["decision_flips"] = flips
            cfg[name]["max_prob_delta"] = round(max(deltas), 4) if deltas else 0.0
        cfg["share_stats"] = dict(getattr(backend, "share_stats", {}))
        res[label] = cfg
        print(f"{label}: exact {cfg['exact']['time']['p50']}ms, "
              f"tree {cfg['tree']['time']['p50']}ms ({cfg['tree']['speedup_vs_exact']}x), "
              f"cache {cfg['cache']['time']['p50']}ms ({cfg['cache']['speedup_vs_exact']}x), "
              f"stats {cfg['share_stats']}", flush=True)

    out = Path(__file__).parent / "e2c_result.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
