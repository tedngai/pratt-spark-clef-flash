"""Probe: why does share_context never engage? Read backend.share_stats."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import transformers


def main() -> None:
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
    long_state = ("The monitoring dashboard flagged elevated error rates on the checkout service and payments began failing. " * 30)

    out = {}
    for name, state in (("short_state", "hello world " * 20), ("long_state", long_state)):
        rt.system_one(state=state, questions=questions, share_context=True)
        out[name] = dict(getattr(backend, "share_stats", {}))

    out["transformers_version"] = transformers.__version__
    try:
        backbone = backend.model.backbone
        out["backbone_model_type"] = getattr(backbone.config, "model_type", "?")
    except Exception as exc:  # noqa: BLE001
        out["backbone_model_type"] = f"unreachable: {exc!r}"

    scmod = next((m for n, m in sys.modules.items() if n.endswith("shared_ctx")), None)
    if scmod is not None and hasattr(scmod, "_supported"):
        out["_supported_source"] = inspect.getsource(scmod._supported)
    if scmod is not None and hasattr(scmod, "auto_shared_tokens"):
        out["_auto_shared_tokens_source"] = inspect.getsource(scmod.auto_shared_tokens)

    Path(__file__).with_name("share_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
