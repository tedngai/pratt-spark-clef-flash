"""Published reference numbers for decision models (do not modify benchmark logic here).

Sources:
- "Introducing Clef" blog (blog.cloudflare.com/clef-decision-models) and the
  Cloudflare/clef + Cloudflare/clef-flash model cards (Jev Decision Index tables).
- Ollama clef-flash page for Decision Index latency rows (median/p95).
- TypeSafe AI blog + docs for Jev pricing/latency claims.
"""

# Jev Decision Index — quality (all values are percent)
# Order of models: clef, clef_flash, jev, diffusiongemma_jev, kev_9b, laya
DECISION_INDEX: dict[str, dict[str, float]] = {
    # benchmark: {model: value}
    "BFCL (case exact acc)": {
        "Clef 27B": 98.47, "Clef-flash 9B": 98.76, "Jev": 95.75,
        "DiffusionGemma Jev": 96.52, "Kev 9B": 94.51, "Laya": 38.13,
    },
    "ToolRet (nDCG@10)": {
        "Clef 27B": 69.19, "Clef-flash 9B": 66.43, "Jev": 65.28,
        "DiffusionGemma Jev": 61.21, "Kev 9B": 64.26, "Laya": 12.69,
    },
    "API-Bank (accuracy)": {
        "Clef 27B": 91.93, "Clef-flash 9B": 93.11, "Jev": 88.19,
        "DiffusionGemma Jev": 83.66, "Kev 9B": 56.30, "Laya": 11.41,
    },
    "Home appliances (case exact acc)": {
        "Clef 27B": 82.95, "Clef-flash 9B": 97.73, "Jev": 52.27,
        "DiffusionGemma Jev": 42.05, "Kev 9B": 25.00, "Laya": 0.00,
    },
    "When2Call (accuracy)": {
        "Clef 27B": 72.37, "Clef-flash 9B": 65.58, "Jev": 80.97,
        "DiffusionGemma Jev": 75.44, "Kev 9B": 49.62, "Laya": 11.94,
    },
    "BANKING77 (macro-F1)": {
        "Clef 27B": 94.20, "Clef-flash 9B": 90.93, "Jev": 79.74,
        "DiffusionGemma Jev": 74.28, "Kev 9B": 84.83, "Laya": 14.29,
    },
    "CLINC150+OOS (macro-F1)": {
        "Clef 27B": 97.43, "Clef-flash 9B": 66.77, "Jev": 89.27,
        "DiffusionGemma Jev": 83.49, "Kev 9B": 79.03, "Laya": 3.19,
    },
    "BRIGHT (nDCG@10)": {
        "Clef 27B": 45.91, "Clef-flash 9B": 39.26, "Jev": 47.52,
        "DiffusionGemma Jev": 42.94, "Kev 9B": 38.53, "Laya": 19.90,
    },
    "Amazon ESCI (macro-F1)": {
        "Clef 27B": 57.48, "Clef-flash 9B": 57.39, "Jev": 55.21,
        "DiffusionGemma Jev": 53.37, "Kev 9B": 49.22, "Laya": 24.40,
    },
    "PhishNChips (accuracy)": {
        "Clef 27B": 79.60, "Clef-flash 9B": 75.05, "Jev": 62.55,
        "DiffusionGemma Jev": 85.35, "Kev 9B": 50.75, "Laya": 50.15,
    },
}

# Decision Index latency rows (ms). Note: Cloudflare-hosted infra, their harness schema mix.
DECISION_INDEX_LATENCY: dict[str, dict[str, float]] = {
    "median latency (ms)": {"Clef 27B": 209.3, "Clef-flash 9B": 38.8, "Jev": 524.1},
    "p95 latency (ms)": {"Clef 27B": 238.6, "Clef-flash 9B": 122.4, "Jev": 536.0},
}

# TypeSafe workflow evals (from the Clef launch blog / model cards, percent)
TYPESAFE_WORKFLOWS: dict[str, dict[str, float]] = {
    "Invoice processing (exact actions)": {"Clef 27B": 64.7, "Clef-flash 9B": 57.1, "Jev": 61.8},
    "Customer service (exact actions)": {"Clef 27B": 76.3, "Clef-flash 9B": 77.0, "Jev": 76.0},
    "Security incidents (exact actions)": {"Clef 27B": 62.9, "Clef-flash 9B": 61.7, "Jev": 61.7},
    "Agent trace observability": {"Clef 27B": 68.5, "Clef-flash 9B": 69.8, "Jev": 71.6},
}

# Our local dataset -> the Decision Index row it corresponds to
DATASET_TO_INDEX_ROW = {
    "banking77": "BANKING77 (macro-F1)",
    "clinc150": "CLINC150+OOS (macro-F1)",
    # ANLI is on the clef (27B) card; values below from the clef-flash HF card table.
}
ANLI_MACRO_F1: dict[str, float] = {
    "Clef 27B": 69.8, "Clef-flash 9B": 59.1, "Jev": 74.8,
    "DiffusionGemma Jev": 66.4, "Kev 9B": 56.3, "Laya": 48.7,
}
