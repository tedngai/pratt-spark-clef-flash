#!/bin/bash
# Launch Decision-2.0-Lux-9B as a second container alongside clef-flash.
# Same image (torch/CUDA stack), transformers upgraded in the writable layer
# only (>=5.17 per the Lux card) — the production clef-flash container is
# untouched. Host networking so the endpoint is 192.168.1.242:8002.
set -e
cd /home/tngai/data/clef-flash

docker rm -f lux9b 2>/dev/null || true

docker run -d --name lux9b \
  --gpus all --ipc host --shm-size 16gb --network host \
  --entrypoint bash \
  -v /home/tngai/data/clef-flash/data/huggingface:/root/.cache/huggingface \
  -v /home/tngai/data/clef-flash:/work \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PORT=8002 \
  -e SERVED_MODEL_NAME=Decision-2.0-Lux-9B \
  clef-flash:latest \
  -c "pip install -q -U 'transformers>=5.17' && exec python /work/lux_server.py"

echo "lux9b container started"
docker logs lux9b 2>&1 | tail -3
