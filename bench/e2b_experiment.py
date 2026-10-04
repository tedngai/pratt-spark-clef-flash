"""E2b: share_context on a LONG shared state (~2000 tokens, 8 questions).

Default policy (tree mode), cache mode, vs the exact path: latency,
decision flips, probability deltas. Saved next to this file.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

MODEL_ID = "vllm-sr/Decision-2.0-Lux-9B"


def sync():
    torch.cuda.synchronize()


def bench(fn, reps: int = 12) -> dict:
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(statistics.median(ts), 1), "mean": round(statistics.mean(ts), 1)}


def main() -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cuda")
    rt = model.runtime

    state = (
        "The monitoring dashboard flagged elevated error rates on the checkout service and "
        "payments began failing for about twelve minutes. The on-call engineer paged the "
        "payments team, who rolled back the last deployment. Error rates returned to baseline "
        "within five minutes. A postmortem was scheduled and the incident was closed the same "
        "evening. The follow-up actions included adding a canary stage and an automatic rollback "
        "trigger tied to the payment error budget. "
    ) * 20

    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Does the message mention issue {i}?",
            "criteria": {"yes": "It is explicitly mentioned", "no": "It is not mentioned"},
        }
        for i in range(8)
    }

    def ask(sc):
        out = rt.system_one(state=state, questions=questions, share_context=sc)
        return out["answers"] if isinstance(out, dict) else out[0]

    for sc in (False, True, {"mode": "cache"}):  # warmup/compile all paths
        ask(sc)

    res: dict = {}
    res["exact"] = {"time": bench(lambda: ask(False), reps=10)}
    ans_exact = ask(False)
    for name, sc in (("tree", True), ("cache", {"mode": "cache"})):
        res[name] = {"time": bench(lambda: ask(sc))}
        ans = ask(sc)
        flips = sum(1 for k in ans_exact if ans_exact[k]["choice"] != ans.get(k, {}).get("choice"))
        deltas = [
            abs(ans_exact[k]["probabilities"][pk] - ans[k]["probabilities"][pk])
            for k in ans_exact
            if k in ans
            for pk in ans_exact[k]["probabilities"]
            if pk in ans[k].get("probabilities", {})
        ]
        res[name]["speedup_vs_exact"] = round(res["exact"]["time"]["p50"] / res[name]["time"]["p50"], 2)
        res[name]["decision_flips"] = flips
        res[name]["max_prob_delta"] = round(max(deltas), 4) if deltas else 0.0

    res["notes"] = {
        "state_repeat_tokens_approx": "state is 20 repeats of ~110-token paragraph (~2200 tokens)",
        "questions": 8,
    }
    out = Path(__file__).parent / "e2b_result.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
