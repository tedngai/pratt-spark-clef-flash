"""Serial clinc150 probe: check for 500s and get clean serial latency."""
from run_bench import load_dataset, questions_for, call_systemone, extract_answer

rows = load_dataset("clinc150")
q = questions_for("clinc150", rows[0])
fails = 0
for row in rows[20:40]:
    try:
        result, cms, sms = call_systemone(
            "http://192.168.1.242:8001", "/v1/systemone", "clef-flash", row["state"], q, retries=1
        )
        a = extract_answer("clinc150", result)
        ok = a["predicted"] == row["label"]
        print(f"{row['id']}: server={sms}ms client={cms:.0f}ms correct={ok} expected={row['label']} got={a['predicted']}")
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(row["id"], "FAILED:", str(e)[:300])
print("serial fails:", fails)
