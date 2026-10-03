#!/bin/bash
# Launch the Lux-9B benchmark detached from any SSH session (survives PC sleep).
cd /home/tngai/data/clef-flash/bench
nohup python3 run_bench.py --dataset all --concurrency 1 --tag lux9b_ \
  --base-url http://127.0.0.1:8002 --model Decision-2.0-Lux-9B \
  > /home/tngai/data/clef-flash/bench/lux_bench.log 2>&1 &
echo "BENCH-STARTED pid $!"
sleep 12
tail -4 /home/tngai/data/clef-flash/bench/lux_bench.log
