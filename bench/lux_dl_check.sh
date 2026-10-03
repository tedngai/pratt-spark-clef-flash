#!/bin/bash
BLOBS=/home/tngai/data/clef-flash/data/huggingface/hub/models--vllm-sr--Decision-2.0-Lux-9B/blobs
echo "== process =="; ps aux | grep '[s]napshot_download' | wc -l
echo "== log =="; tail -c 400 /home/tngai/data/lux_download.log | tr '\r' '\n' | tail -2
echo "== blobs =="; du -sh $BLOBS 2>/dev/null; ls $BLOBS 2>/dev/null | wc -l
echo "== incomplete =="; du -bc $BLOBS/*.incomplete 2>/dev/null | tail -1
echo "== cache total =="; du -sh /home/tngai/data/clef-flash/data/huggingface
