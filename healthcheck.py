"""Container healthcheck: exit 0 only when the model has finished loading."""

from __future__ import annotations

import os
import sys
import urllib.request

port = os.environ.get("PORT", "8001")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:  # noqa: BLE001
    sys.exit(1)
