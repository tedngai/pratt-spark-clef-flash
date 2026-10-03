"""Download BANKING77, CLINC150+OOS, and ANLI test sets into data/*.jsonl.

Sources:
- BANKING77: original PolyAI GitHub tarball (hub datasets-server viewer is broken for it)
- CLINC150+OOS: HF datasets-server (clinc/clinc_oos, config 'plus', split 'test', 5,500 rows)
- ANLI: HF datasets-server (facebook/anli, config 'plain_text', test_r1/r2/r3)

Output rows are normalized to:
  {"dataset": str, "id": str, "state": {...}, "label": str, "meta": {...}}
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SERVER = "https://datasets-server.huggingface.co/rows"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path.name}")


def fetch_banking77() -> None:
    out = DATA / "banking77.jsonl"
    if out.exists():
        print("banking77 already fetched")
        return
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    print(f"downloading BANKING77 from {base}")
    csv_bytes = urllib.request.urlopen(f"{base}/test.csv", timeout=120).read().decode("utf-8")
    cats = json.loads(urllib.request.urlopen(f"{base}/categories.json", timeout=30).read().decode("utf-8"))
    cat_set = set(cats) if isinstance(cats, list) else set(cats.keys())

    rows = []
    seen_labels = set()
    reader = csv.DictReader(io.StringIO(csv_bytes))
    for i, rec in enumerate(reader):
        text = rec["text"].strip()
        label = rec["category"].strip()
        assert label in cat_set, f"unknown label {label!r}"
        seen_labels.add(label)
        rows.append(
            {
                "dataset": "banking77",
                "id": f"banking77-{i}",
                "state": {"customer_message": text},
                "label": label,
                "meta": {"label_text": label},
            }
        )
    assert len(seen_labels) == 77, f"expected 77 categories, saw {len(seen_labels)}"
    write_jsonl(out, rows)


def fetch_rows_page(dataset: str, config: str, split: str, offset: int, length: int = 100) -> dict:
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            r = requests.get(
                SERVER,
                params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length},
                timeout=60,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} Server Error")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            print(f"  rows(offset={offset}) attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"datasets-server failed after retries: {last_exc}")


def class_label_names(dataset: str, config: str, split: str, column: str) -> list[str]:
    """Resolve ClassLabel names via the parquet info endpoint, falling back to first-rows."""
    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}/info", timeout=60)
        r.raise_for_status()
        feats = r.json()["features"]
        if isinstance(feats, list):
            for feat in feats:
                if feat["name"] == column and isinstance(feat["type"], dict) and "names" in feat["type"]:
                    return feat["type"]["names"]
        elif isinstance(feats, dict) and column in feats:
            info = feats[column]
            names = info.get("names") or info.get("_type")
            if isinstance(names, list):
                return names
    except Exception as exc:  # noqa: BLE001
        print(f"  info endpoint failed ({exc}); falling back to first-rows")
    first = fetch_rows_page(dataset, config, split, 0, 1)
    for feat in first["features"]:
        if feat["name"] == column and isinstance(feat["type"], dict) and "names" in feat["type"]:
            return feat["type"]["names"]
    raise RuntimeError(f"could not resolve ClassLabel names for {column}")


def fetch_parquet(dataset: str, config: str, split: str) -> "pyarrow.Table":
    r = requests.get(f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}", timeout=60)
    r.raise_for_status()
    urls = r.json()
    tables = []
    for url in urls:
        raw = urllib.request.urlopen(url, timeout=300).read()
        tables.append(pq.read_table(io.BytesIO(raw)))
    return pa.concat_tables(tables)


def fetch_clinc150() -> None:
    out = DATA / "clinc150.jsonl"
    if out.exists():
        print("clinc150 already fetched")
        return
    dataset, config, split = "clinc/clinc_oos", "plus", "test"
    names = class_label_names(dataset, config, split, "intent")
    assert len(names) == 151, f"expected 151 intent labels, got {len(names)}"

    table = fetch_parquet(dataset, config, split)
    texts = table.column("text").to_pylist()
    intents = table.column("intent").to_pylist()
    print(f"clinc150: {len(texts)} test rows, 151 intents")
    rows = [
        {
            "dataset": "clinc150",
            "id": f"clinc150-{i}",
            "state": {"user_utterance": text},
            "label": names[intent],
            "meta": {"label_idx": intent},
        }
        for i, (text, intent) in enumerate(zip(texts, intents))
    ]
    write_jsonl(out, rows)


def fetch_anli() -> None:
    out = DATA / "anli.jsonl"
    if out.exists():
        print("anli already fetched")
        return
    dataset, config = "facebook/anli", "plain_text"
    label_names = ["entailment", "neutral", "contradiction"]

    rows = []
    idx = 0
    for rnd in ("test_r1", "test_r2", "test_r3"):
        table = fetch_parquet(dataset, config, rnd)
        premises = table.column("premise").to_pylist()
        hypotheses = table.column("hypothesis").to_pylist()
        labels = table.column("label").to_pylist()
        print(f"  {rnd}: {len(premises)} rows")
        for premise, hypothesis, label in zip(premises, hypotheses, labels):
            rows.append(
                {
                    "dataset": "anli",
                    "id": f"anli-{rnd}-{idx}",
                    "state": {"premise": premise, "hypothesis": hypothesis},
                    "label": label_names[label],
                    "meta": {"round": rnd},
                }
            )
            idx += 1
    write_jsonl(out, rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "banking77"):
        fetch_banking77()
    if which in ("all", "clinc150"):
        fetch_clinc150()
    if which in ("all", "anli"):
        fetch_anli()
    print("done")
