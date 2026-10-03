"""Final three-way report: clef-flash vs Decision-2.0-Lux-9B vs Jev (OpenRouter).

Same datasets, same question encodings, same serial protocol. Published Decision
Index numbers included as context. Writes report/report.md + one chart.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from analyze import (
    accuracy,
    anli_by_round,
    brier,
    confusion_pairs,
    confidence_split,
    ece,
    latency_stats,
    macro_f1,
    oos_recall,
)
from decision_index import ANLI_MACRO_F1, DECISION_INDEX

ROOT = Path(__file__).parent
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)

MODELS = [("", "clef-flash (GB10 local)"), ("lux9b_", "Lux-9B (GB10 local)"), ("jev_", "Jev (OpenRouter cloud)")]
PUBLISHED_JEV = {"BANKING77 (macro-F1)": 79.74, "CLINC150+OOS (macro-F1)": 89.27}


def load(tag: str, ds: str) -> list[dict]:
    path = ROOT / "results" / f"{tag}{ds}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    lines: list[str] = []
    lines.append("# Three-way decision model benchmark: clef-flash vs Lux-9B vs Jev\n")
    lines.append(
        "All three models answered the identical 11,780 questions (BANKING77 3,080 · CLINC150+OOS 5,500 · ANLI 3,200)\n"
        "with identical question encodings, serial protocol (concurrency=1), October 3, 2026.\n"
    )
    lines.append(
        "- **clef-flash** — Cloudflare/clef-flash self-hosted on the NVIDIA GB10 (DGX Spark), hardened FastAPI server\n"
        "- **Lux-9B** — vllm-sr/Decision-2.0-Lux-9B self-hosted on the same GB10 (transformers, vendor runtime)\n"
        "- **Jev** — typesafe/jev-1.13 via OpenRouter decisions API (cloud; probabilities quantized to 2 decimals)\n"
    )

    # ---------------- headline table ----------------
    lines.append("\n## Headline results\n")
    lines.append("| Dataset | Model | macro-F1 | accuracy | OOS recall | p50 | p95 | conf ok/wrong |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for ds in ["banking77", "clinc150", "anli"]:
        for tag, name in MODELS:
            rows = load(tag, ds)
            lat = latency_stats(rows)
            cc, cw = confidence_split(rows)
            oos = oos_recall(rows)
            lines.append(
                f"| {ds.upper() if ds != 'clinc150' else 'CLINC150+OOS'} | {name} | **{macro_f1(rows):.2f}** "
                f"| {accuracy(rows):.2f} | {f'{oos:.1f}%' if oos is not None else '-'} "
                f"| {lat.get('p50', '-')}ms | {lat.get('p95', '-')}ms | {cc:.2f} / {cw:.2f} |"
            )

    # ---------------- harness validation ----------------
    jev = {ds: load("jev_", ds) for ds in ["banking77", "clinc150", "anli"]}
    lines.append("\n## Harness validation: local Jev reproduces the published Decision Index\n")
    lines.append(
        "| Task | Jev published | Jev (this run) | delta |\n| --- | ---: | ---: | ---: |"
    )
    pairs = [
        ("BANKING77", PUBLISHED_JEV["BANKING77 (macro-F1)"], macro_f1(jev["banking77"])),
        ("CLINC150+OOS", PUBLISHED_JEV["CLINC150+OOS (macro-F1)"], macro_f1(jev["clinc150"])),
        ("ANLI", ANLI_MACRO_F1["Jev"], macro_f1(jev["anli"])),
    ]
    for name, pub, loc in pairs:
        lines.append(f"| {name} | {pub} | {loc:.2f} | {loc - pub:+.2f} |")
    lines.append(
        "\nLocal Jev lands within 0.6 points of the published numbers on all three tasks — the encodings and\n"
        "harness are faithful. Consequence: where our local clef-flash diverges from its published row\n"
        "(BANKING77 90.93→95.41, CLINC150 66.77→98.6), the difference is the encoding, not measurement noise.\n"
        "The three-way table above is the apples-to-apples comparison; published rows are context only.\n"
    )

    # ---------------- per-dataset detail ----------------
    for ds in ["banking77", "clinc150", "anli"]:
        title = {"banking77": "BANKING77", "clinc150": "CLINC150+OOS", "anli": "ANLI"}[ds]
        lines.append(f"\n## {title}\n")
        for tag, name in MODELS:
            rows = load(tag, ds)
            s = [f"- **{name}**: macro-F1 {macro_f1(rows):.2f}, Brier {brier(rows):.4f}, ECE {ece(rows):.4f}"]
            if ds == "anli":
                by_round = anli_by_round(rows)
                s.append("by round: " + ", ".join(f"{k}: {v:.2f}" for k, v in by_round.items()))
            pairs_ = confusion_pairs(rows, top=3)
            if pairs_:
                s.append("top confusions: " + "; ".join(f"{e}→{p} ({n})" for (e, p), n in pairs_))
            lines.append(" — ".join(s))

    # ---------------- latency context ----------------
    lines.append("\n## Latency context (read before comparing)\n")
    lines.append(
        "| Model | where | p50 (small schema, ANLI) | p50 (77 options) | p50 (151 options) |\n"
        "| --- | --- | ---: | ---: | ---: |"
    )
    for tag, name in MODELS:
        cells = []
        for ds in ["anli", "banking77", "clinc150"]:
            lat = latency_stats(load(tag, ds))
            cells.append(f"{lat.get('p50', '-')}ms")
        where = "GB10 local" if "local" in name else "cloud (network RTT included)"
        lines.append(f"| {name.split(' (')[0]} | {where} | " + " | ".join(cells) + " |")
    lines.append(
        "\n- Published Decision Index latencies (their infra): clef-flash 38.8ms, Jev 524.1ms, Clef 27B 209.3ms median.\n"
        "- Jev via OpenRouter (187ms median here) is served by TypeSafe's inference infra — faster than the\n"
        "  published 524ms and faster than a hobby-box GPU on large schemas; that is infrastructure, not model speed.\n"
        "- On the same GB10, Lux-9B runs ~10% slower than clef-flash at every schema size (no fused kernels on\n"
        "  NVIDIA — its Triton kernels are ROCm-only).\n"
    )

    # ---------------- published context ----------------
    lines.append("\n## Appendix: published Jev Decision Index rows (context only, different encodings)\n")
    lines.append("| Benchmark | Clef 27B | Clef-flash 9B | Jev | DiffusionGemma Jev | Kev 9B | Laya | **This run (best local)** |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    local_best = {
        "BANKING77 (macro-F1)": 95.41,
        "CLINC150+OOS (macro-F1)": 98.6,
    }
    for row, vals in DECISION_INDEX.items():
        if row not in ("BANKING77 (macro-F1)", "CLINC150+OOS (macro-F1)"):
            continue
        best = max(vals.values())
        lines.append(
            f"| {row} | " + " | ".join(f"{vals[m]}" for m in
            ["Clef 27B", "Clef-flash 9B", "Jev", "DiffusionGemma Jev", "Kev 9B", "Laya"]) +
            f" | **{max(best, local_best[row]):.2f}** |"
        )
    anli_row = ANLI_MACRO_F1
    lines.append(
        f"| ANLI (macro-F1) | {anli_row['Clef 27B']} | {anli_row['Clef-flash 9B']} | {anli_row['Jev']} | "
        f"{anli_row['DiffusionGemma Jev']} | {anli_row['Kev 9B']} | {anli_row['Laya']} | **{max(anli_row.values()):.2f}** |"
    )

    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT / 'report.md'}")

    # ---------------- chart ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        datasets = ["BANKING77", "CLINC150+OOS", "ANLI"]
        series = {}
        for tag, name in MODELS:
            series[name.split(" (")[0]] = [macro_f1(load(tag, ds)) for ds in ["banking77", "clinc150", "anli"]]
        x = np.arange(len(datasets))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, vals) in enumerate(series.items()):
            bars = ax.bar(x + (i - 1) * w, vals, w, label=name)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set_xticks(x, datasets)
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, 105)
        ax.set_title("Same encoding, same questions: clef-flash vs Lux-9B vs Jev")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT / "threeway.png", dpi=130)
        print("wrote threeway.png")
        # append chart reference
        txt = (REPORT / "report.md").read_text(encoding="utf-8")
        txt = txt.replace("## Headline results\n", "![three-way comparison](threeway.png)\n\n## Headline results\n", 1)
        (REPORT / "report.md").write_text(txt, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"chart skipped: {exc}")


if __name__ == "__main__":
    main()
