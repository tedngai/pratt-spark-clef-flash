#!/bin/bash
SNAP=/home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/snapshots/78bf3c03d9147aeb30b641edfe0e30ed04887ca5
echo "== micro_batches def =="
find "$SNAP" -name '*.py' -exec grep -Hn "def micro_batches" {} \;
find "$SNAP" -name '*.py' -exec grep -Hn -A 14 "def micro_batches" {} \; | head -20
echo
echo "== batch_tokens / share_context defaults =="
find "$SNAP" -name '*.py' -exec grep -Hn "batch_tokens\s*[:=]\|share_context\s*[:=]\s*\|_BATCH_TOKENS\|forward token budget" {} \; | grep -v "def \|self.batch_tokens)" | head -12
