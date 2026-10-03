"""Compute benchmark metrics from results/*.jsonl and build report/report.md.

Metrics per dataset:
- accuracy, macro-F1 (over ground-truth classes)
- CLINC150: OOS recall (out-of-scope detection rate)
- ANLI: macro-F1 per adversarial round
- latency: server-side p50/p95/p99 + throughput (serial harness)
- calibration: mean confidence on correct vs wrong, Brier score, ECE (10 bins)

Published Decision Index rows are merged into the comparison table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_index import ANLI_MACRO_F1, DATASET_TO_INDEX_ROW, DECISION_INDEX

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)

DATASETS = ["anli", "banking77", "clinc150"]
DISPLAY = {"anli": "ANLI", "banking77": "BANKING77", "clinc150": "CLINC150+OOS"}


def load(dataset: str) -> list[dict]:
    path = RESULTS / f"{dataset}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def macro_f1(rows: list[dict]) -> float:
    """Macro-averaged F1 over the classes appearing in the ground truth."""
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        exp, pred = r["expected"], r["predicted"]
        if pred == exp:
            by_class[exp]["tp"] += 1
        else:
            by_class[exp]["fn"] += 1
            by_class[pred]["fp"] += 1
    f1s = []
    for cls, c in by_class.items():
        denom = 2 * c["tp"] + c["fp"] + c["fn"]
        f1s.append(2 * c["tp"] / denom if denom else 0.0)
    return 100 * sum(f1s) / len(f1s) if f1s else 0.0


def accuracy(rows: list[dict]) -> float:
    return 100 * sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def oos_recall(rows: list[dict]) -> float | None:
    oos = [r for r in rows if r["expected"] == "oos"]
    if not oos:
        return None
    return 100 * sum(r["correct"] for r in oos) / len(oos)


def brier(rows: list[dict]) -> float:
    """Multiclass Brier score using the recorded per-option probabilities."""
    total, n = 0.0, 0
    for r in rows:
        probs = r.get("probabilities") or {}
        if not probs:
            continue
        expected_prob = probs.get(r["expected"], 0.0)
        # sum over options: (p_o - y_o)^2 ; y is 1 for expected option
        total += sum((p - (1.0 if opt == r["expected"] else 0.0)) ** 2 for opt, p in probs.items())
        n += 1
    return total / n if n else float("nan")


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error over the top-choice confidence."""
    confidences = []
    for r in rows:
        c = r.get("confidence")
        if c is None:
            probs = r.get("probabilities") or {}
            c = max(probs.values()) if probs else None
        if c is not None:
            confidences.append((c, 1.0 if r["correct"] else 0.0))
    if not confidences:
        return float("nan")
    total = len(confidences)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, o) for c, o in confidences if lo <= c < hi or (b == bins - 1 and c == hi)]
        if bucket:
            acc = sum(o for _, o in bucket) / len(bucket)
            conf = sum(c for c, _ in bucket) / len(bucket)
            err += len(bucket) / total * abs(acc - conf)
    return err


def latency_stats(rows: list[dict]) -> dict:
    server = sorted(r["server_ms"] for r in rows if r.get("server_ms"))
    if not server:
        # cloud APIs (OpenRouter) don't report server timing — fall back to
        # client-observed round-trip
        server = sorted(r["client_ms"] for r in rows if r.get("client_ms"))
    if not server:
        return {}
    def pct(q):
        return round(server[min(len(server) - 1, int(q / 100 * len(server)))], 1)
    return {
        "p50": pct(50), "p95": pct(95), "p99": pct(99),
        "mean": round(sum(server) / len(server), 1), "n": len(server),
    }


def anli_by_round(rows: list[dict]) -> dict[str, float]:
    rounds = defaultdict(list)
    for r in rows:
        rounds[r.get("meta", {}).get("round", "?")].append(r)
    return {rnd: macro_f1(rs) for rnd, rs in sorted(rounds.items())}


def confusion_pairs(rows: list[dict], top: int = 5) -> list[tuple[str, str, int]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if not r["correct"]:
            counts[(r["expected"], r["predicted"])] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:top]


def confidence_split(rows: list[dict]) -> tuple[float, float]:
    cs = [r.get("confidence") or (max((r.get("probabilities") or {}).values()) if r.get("probabilities") else None) for r in rows]
    correct = [c for c, r in zip(cs, rows) if c is not None and r["correct"]]
    wrong = [c for c, r in zip(cs, rows) if c is not None and not r["correct"]]
    return (
        sum(correct) / len(correct) if correct else float("nan"),
        sum(wrong) / len(wrong) if wrong else float("nan"),
    )


def main() -> None:
    stats: dict[str, dict] = {}
    for ds in DATASETS:
        rows = load(ds)
        if not rows:
            continue
        s: dict = {
            "n": len(rows),
            "accuracy": round(accuracy(rows), 2),
            "macro_f1": round(macro_f1(rows), 2),
            "latency": latency_stats(rows),
            "brier": round(brier(rows), 4),
            "ece": round(ece(rows), 4),
        }
        conf_c, conf_w = confidence_split(rows)
        s["conf_correct"] = round(conf_c, 3)
        s["conf_wrong"] = round(conf_w, 3)
        oosr = oos_recall(rows) if ds == "clinc150" else None
        if oosr is not None:
            s["oos_recall"] = round(oosr, 2)
        if ds == "anli":
            s["by_round"] = {k: round(v, 2) for k, v in anli_by_round(rows).items()}
        stats[ds] = s
        print(ds, json.dumps(s))

    if not stats:
        raise SystemExit("no results found; run run_bench.py first")

    charts(stats)
    write_report(stats)
    print(f"\nwrote {REPORT/'report.md'}")


def charts(stats: dict) -> None:
    # 1. macro-F1 comparison bar chart: local vs published rows
    fig, ax = plt.subplots(figsize=(11, 5))
    datasets = [ds for ds in DATASETS if ds in stats]
    width = 0.13
    models = ["Clef 27B", "Clef-flash 9B", "Jev", "DiffusionGemma Jev", "Kev 9B", "Laya"]
    colors = ["#f5a623", "#f8e71c", "#4a90d9", "#7ed321", "#9013fe", "#9b9b9b"]
    x = range(len(datasets))
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            if ds == "anli":
                vals.append(ANLI_MACRO_F1[model])
            else:
                vals.append(DECISION_INDEX[DATASET_TO_INDEX_ROW[ds]][model])
        bars = ax.bar([xi + i * width - 2.5 * width for xi in x], vals, width, label=model, color=colors[i])
        # highlight our local run with a marker on the Clef-flash bar
        if model == "Clef-flash 9B":
            for xi, (bar, ds) in enumerate(zip(bars, datasets)):
                local = stats[ds]["macro_f1"]
                ax.plot(xi + i * width - 2.5 * width, local, marker="D", color="black", markersize=7, zorder=5)
    for xi, ds in enumerate(datasets):
        ax.text(xi + 1 * width - 2.5 * width, stats[ds]["macro_f1"] + 1.5, f"{stats[ds]['macro_f1']:.1f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[ds] for ds in datasets])
    ax.set_ylabel("macro-F1 (%)")
    ax.set_title("Local clef-flash (black diamond = our run) vs published Decision Index")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(REPORT / "macro_f1_comparison.png", dpi=150)

    # 2. reliability diagrams per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        rows = load(ds)
        bins = 10
        accs, confs, fracs = [], [], []
        cs = [(r.get("confidence") or max((r.get("probabilities") or {}).values()), r["correct"]) for r in rows]
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            bucket = [(c, o) for c, o in cs if lo <= c < hi or (b == bins - 1 and c == hi)]
            if bucket:
                fracs.append(len(bucket) / len(cs))
                confs.append((b + 0.5) / bins)
                accs.append(sum(o for _, o in bucket) / len(bucket))
            else:
                fracs.append(0)
                confs.append((b + 0.5) / bins)
                accs.append(0)
        ax.bar(confs, fracs, width=0.09, alpha=0.35, color="#4a90d9", label="confidence mass")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.plot(confs, accs, "o-", color="#d0021b", label="accuracy")
        ax.set_title(f"{DISPLAY[ds]} (ECE={stats[ds]['ece']:.3f})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("fraction / accuracy")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT / "reliability.png", dpi=150)


def write_report(stats: dict) -> None:
    lines: list[str] = []
    lines.append("# Clef-flash local benchmark vs Jev and other decision models\n")
    lines.append(
        f"Endpoint: `http://192.168.1.242:8001` (Cloudflare/clef-flash, NVIDIA GB10 / DGX Spark), "
        f"serial requests via `POST /v1/systemone`.\n"
    )
    lines.append(
        "Question encodings are our own (label names as criteria); published numbers come from "
        "Cloudflare's Jev Decision Index harness, so treat cross-harness deltas as approximate. "
        "Local numbers are exact for this endpoint and this encoding.\n"
    )

    # summary table
    lines.append("## Headline results (local run)\n")
    lines.append("| Dataset | n | macro-F1 | accuracy | OOS recall | server p50 | server p95 | server p99 | throughput* |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds in DATASETS:
        if ds not in stats:
            continue
        s = stats[ds]
        lat = s.get("latency", {})
        rps = f"{1000 / lat['mean']:.1f} req/s" if lat.get("mean") else "-"
        lines.append(
            f"| {DISPLAY[ds]} | {s['n']} | **{s['macro_f1']}** | {s['accuracy']} | "
            f"{s.get('oos_recall', '-')} | {lat.get('p50','-')} ms | {lat.get('p95','-')} ms | "
            f"{lat.get('p99','-')} ms | {rps} |"
        )
    lines.append("\n*serial harness; the GPU processes one decision at a time (see latency notes).\n")

    # comparison with published
    lines.append("## vs published Decision Index (macro-F1)\n")
    lines.append("| Model | ANLI | BANKING77 | CLINC150+OOS |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model in ["Clef 27B", "Clef-flash 9B", "Jev", "DiffusionGemma Jev", "Kev 9B", "Laya"]:
        vals = [ANLI_MACRO_F1[model]]
        for ds in ["banking77", "clinc150"]:
            vals.append(DECISION_INDEX[DATASET_TO_INDEX_ROW[ds]][model])
        lines.append(f"| {model} (published) | {vals[0]} | {vals[1]} | {vals[2]} |")
    local_row = ["**clef-flash (this run)**"]
    for ds in ["anli", "banking77", "clinc150"]:
        local_row.append(f"**{stats[ds]['macro_f1']}**" if ds in stats else "-")
    lines.append("| " + " | ".join(local_row) + " |")
    lines.append(
        "\nANLI here is the 3-way task (entailment/neutral/contradiction) scored as macro-F1, "
        "matching the published table's metric.\n"
    )

    # per-dataset detail
    for ds in DATASETS:
        if ds not in stats:
            continue
        s = stats[ds]
        lines.append(f"\n## {DISPLAY[ds]}\n")
        if "by_round" in s:
            lines.append("Per adversarial round (macro-F1): " + ", ".join(f"{k}: {v}" for k, v in s["by_round"].items()))
        if "oos_recall" in s:
            lines.append(f"OOS detection recall: **{s['oos_recall']}%** (1,000 out-of-scope utterances)")
        lines.append(
            f"Calibration: mean confidence {s['conf_correct']} when correct vs {s['conf_wrong']} when wrong; "
            f"Brier {s['brier']}; ECE {s['ece']}.\n"
        )
        rows = load(ds)
        pairs = confusion_pairs(rows)
        if pairs:
            lines.append("Top confusions (expected -> predicted, count):")
            for (exp, pred), n in pairs:
                lines.append(f"- {exp} -> {pred} ({n})")

    # latency + published latency context
    lines.append("\n## Latency context\n")
    lines.append("| Source | median | p95 |")
    lines.append("| --- | ---: | ---: |")
    lines.append("| Decision Index (Cloudflare infra): Clef-flash 9B | 38.8 ms | 122.4 ms |")
    lines.append("| Decision Index (Cloudflare infra): Jev | 524.1 ms | 536.0 ms |")
    lines.append("| Decision Index (Cloudflare infra): Clef 27B | 209.3 ms | 238.6 ms |")
    lat_anli = stats.get("anli", {}).get("latency", {})
    if lat_anli:
        lines.append(
            f"| **This endpoint, small schema (~275 tok, 3 options)** | {lat_anli['p50']} ms | {lat_anli['p95']} ms |"
        )
    for ds in ("banking77", "clinc150"):
        lat = stats.get(ds, {}).get("latency", {})
        if lat:
            lines.append(
                f"| This endpoint, {'77' if ds == 'banking77' else '151'}-option schema | {lat['p50']} ms | {lat['p95']} ms |"
            )
    lines.append(
        "\nNotes: published Decision Index latencies use their own (smaller) schema mix. "
        "The joint schema head scales with option count, so large classifications dominate latency. "
        "Concurrency does not increase aggregate throughput on this single-GPU server (requests serialize on the model).\n"
    )

    if (REPORT / "macro_f1_comparison.png").exists():
        lines.append("![macro-F1 comparison](macro_f1_comparison.png)\n")
    if (REPORT / "reliability.png").exists():
        lines.append("![reliability diagrams](reliability.png)\n")

    # appendix: published tables
    lines.append("\n## Appendix: full published Jev Decision Index\n")
    lines.append("| Benchmark | " + " | ".join(["Clef 27B", "Clef-flash 9B", "Jev", "DiffusionGemma Jev", "Kev 9B", "Laya"]) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * 6) + " |")
    for bench, row in DECISION_INDEX.items():
        lines.append(f"| {bench} | " + " | ".join(str(row[m]) for m in ["Clef 27B", "Clef-flash 9B", "Jev", "DiffusionGemma Jev", "Kev 9B", "Laya"]) + " |")

    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
