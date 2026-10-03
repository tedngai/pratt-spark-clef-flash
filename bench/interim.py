"""Interim metrics on completed datasets."""
import json

from analyze import accuracy, confidence_split, latency_stats, macro_f1

for ds in ["anli", "banking77"]:
    rows = [json.loads(l) for l in open(f"results/{ds}.jsonl", encoding="utf-8")]
    lat = latency_stats(rows)
    cc, cw = confidence_split(rows)
    print(
        f"{ds}: n={len(rows)} macro-F1={macro_f1(rows):.2f} acc={accuracy(rows):.2f} "
        f"p50={lat['p50']}ms p95={lat['p95']}ms conf_correct={cc:.3f} conf_wrong={cw:.3f}"
    )
