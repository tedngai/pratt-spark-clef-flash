"""Preview: Lux-9B vs clef-flash on completed datasets."""

import json

from analyze import accuracy, confidence_split, latency_stats, macro_f1

print(f"{'dataset':<12} {'model':<12} {'n':>6} {'macro-F1':>9} {'acc':>7} {'p50 ms':>8} {'conf ok/wrong':>14}")
for ds in ["banking77", "clinc150", "anli"]:
    for tag, name in [("", "clef-flash"), ("lux9b_", "Lux-9B"), ("jev_", "Jev")]:
        path = f"results/{tag}{ds}.jsonl"
        try:
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        except FileNotFoundError:
            continue
        lat = latency_stats(rows)
        cc, cw = confidence_split(rows)
        print(f"{ds:<12} {name:<12} {len(rows):>6} {macro_f1(rows):>9.2f} {accuracy(rows):>7.2f} {lat['p50']:>8.0f} {cc:.2f} / {cw:.2f}")
    print()
