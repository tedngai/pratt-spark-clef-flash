#!/bin/bash
# E2b one-shot: stop lux server, run long-state share_context experiment, restart server.
set -e
docker stop lux9b
docker run --rm --name lux_exp \
  --gpus all --ipc host --shm-size 16gb --network host \
  --entrypoint bash \
  -v /home/tngai/data/clef-flash/data/huggingface:/root/.cache/huggingface \
  -v /home/tngai/data/clef-flash:/work \
  -e HF_HOME=/root/.cache/huggingface \
  clef-flash:latest \
  -c "pip install -q -U 'transformers>=5.17' && python /work/bench/e2b_experiment.py" \
  2>&1 | grep -v 'FutureWarning\|warnings.warn\|Loading weights\|Fetching\|it/s\]\|transformers\]' || true
echo "== restarting lux9b server =="
docker start lux9b
