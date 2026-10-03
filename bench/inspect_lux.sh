#!/bin/bash
# Inspect the Decision-2.0-Lux-9B snapshot code for batching surface.
SNAP=$(ls -d /home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/snapshots/* | head -1)
echo "== files =="
ls -la "$SNAP" | awk '{print $5, $9}'
echo
echo "== py files =="
find "$SNAP" -name '*.py' -exec ls -la {} \;
echo
echo "== batch/collate mentions =="
grep -rn "def system_one\|collate\|def forward\|batch" "$SNAP" --include='*.py' | grep -v '^Binary' | head -40
