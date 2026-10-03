#!/bin/bash
echo "== containers =="
docker ps --format '{{.Names}} {{.Status}}' | head -5
echo "== lux9b state =="
docker inspect lux9b --format '{{.State.Status}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}}' 2>/dev/null
echo "== memory =="
free -g | head -2
echo "== dmesg oom =="
dmesg -T 2>/dev/null | grep -i 'out of memory\|oom' | tail -3 || echo "(no dmesg access)"
echo "== python proc in lux9b =="
docker top lux9b 2>/dev/null | head -5
