#!/bin/bash
SNAP=/home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/snapshots/78bf3c03d9147aeb30b641edfe0e30ed04887ca5
echo "== fast_kernels.py header =="
find "$SNAP/decision2" -name 'fast_kernels.py' -exec head -40 {} \;
echo
echo "== collate() in decision_model.py =="
find "$SNAP" -path '*dev2model*' -name 'decision_model.py' -exec sed -n '150,215p' {} \;
echo
echo "== qwen.py system_one body (rows/batch usage) =="
find "$SNAP/decision2" -name 'qwen.py' -exec sed -n '295,360p' {} \;
