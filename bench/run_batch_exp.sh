#!/bin/bash
# Run the batching experiment with exclusive GPU: stop the lux server, run
# one-shot container, restart the server after.
set -e
docker stop lux9b
docker run --rm --name lux_exp \
  --gpus all --ipc host --shm-size 16gb --network host \
  --entrypoint bash \
  -v /home/tngai/data/clef-flash/data/huggingface:/root/.cache/huggingface \
  -v /home/tngai/data/clef-flash:/work \
  -e HF_HOME=/root/.cache/huggingface \
  clef-flash:latest \
  -c "pip install -q -U 'transformers>=5.17' && python /work/bench/batch_experiment.py" \
  2>&1 | grep -v 'Loading weights\|Fetching\|it/s\]\|FutureWarning\|warnings.warn\|^\s*$' || true
echo "== restarting lux9b server =="
docker start lux9b
