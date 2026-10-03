#!/bin/bash
# Launch the Jev benchmark (OpenRouter decisions API) detached from any SSH session.
cd /home/tngai/data/clef-flash/bench
nohup bash -c 'export OPENROUTER_API_KEY=$(tr -d "\r\n " < /home/tngai/data/clef-flash/bench/.openrouter); python3 run_bench.py --dataset all --concurrency 1 --tag jev_ --base-url https://openrouter.ai --endpoint /api/alpha/decisions --model typesafe/jev-1.13 --auth-env OPENROUTER_API_KEY --skip-health' \
  > /home/tngai/data/clef-flash/bench/jev_bench.log 2>&1 &
echo "JEV-BENCH-STARTED pid $!"
sleep 10
tail -3 /home/tngai/data/clef-flash/bench/jev_bench.log
