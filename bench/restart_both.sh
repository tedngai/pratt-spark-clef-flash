#!/bin/bash
# Free the caching-allocator memory clef-flash is squatting on, then bring both
# models up side by side: clef-flash (8001) fresh + lux9b (8002).
set -e
echo "== restarting clef-flash to release allocator cache =="
docker restart clef-flash
cd /home/tngai/data/clef-flash
for i in $(seq 1 60); do
  H=$(curl -s -m 5 http://127.0.0.1:8001/health || true)
  case "$H" in *'"ready":true'*) echo "clef-flash READY after ~$((i * 5))s"; break ;; esac
  sleep 5
done
case "$H" in *'"ready":true'*) ;; *) echo "clef-flash not ready: $H"; exit 1 ;; esac

echo "== starting lux9b =="
bash /home/tngai/data/clef-flash/start_lux.sh
bash /tmp/wait_lux.sh
echo "== both endpoints =="
curl -s -m 5 http://127.0.0.1:8001/health; echo
curl -s -m 5 http://127.0.0.1:8002/health; echo
