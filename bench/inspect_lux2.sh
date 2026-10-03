#!/bin/bash
SNAP=/home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/snapshots/78bf3c03d9147aeb30b641edfe0e30ed04887ca5
echo "== public API surface (api.py) =="
find "$SNAP/decision2" -name 'api.py' -exec grep -n "def \|class " {} \; | head -40
echo
echo "== system_one / batch entry points across package =="
find "$SNAP" -name '*.py' -exec grep -Hn "def system_one\|def batch\|def collate\|def infer_batch\|batch_size\|List\[Record\]\|list\[Record\]" {} \; | head -30
echo
echo "== shared_ctx.py (top of file) =="
find "$SNAP/decision2" -name 'shared_ctx.py' -exec head -60 {} \;
