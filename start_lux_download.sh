#!/bin/bash
# Download Lux-9B into the shared HF cache volume, detached from the session.
nohup docker exec clef-flash python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("vllm-sr/Decision-2.0-Lux-9B"))' > /home/tngai/data/lux_download.log 2>&1 &
echo "DOWNLOAD-STARTED pid $!"
