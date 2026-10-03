#!/bin/bash
echo "== top host processes by RSS =="
ps aux --sort=-rss | head -8 | awk '{printf "%-10s %8.1fGB %s\n", $1, $6/1048576, $11}'
echo "== docker stats =="
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null
echo "== nvidia-smi processes =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -8
