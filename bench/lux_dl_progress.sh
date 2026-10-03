#!/bin/bash
# Report Lux-9B download progress (bytes across .incomplete blobs, twice, 20s apart)
BLOBS=/home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/blobs
A=$(du -bc $BLOBS/*.incomplete 2>/dev/null | tail -1 | cut -f1)
sleep 20
B=$(du -bc $BLOBS/*.incomplete 2>/dev/null | tail -1 | cut -f1)
RATE=$(( (B - A) / 20 / 1048576 ))
echo "bytes: $A -> $B  | rate ~${RATE} MB/s | eta for 16GB: $(( (17179869184 - B) / (RATE * 1048576 + 1) ))s"
