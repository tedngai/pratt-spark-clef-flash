#!/bin/bash
# Wait for the lux9b endpoint to be ready (max ~8 min); abort on load error.
for i in $(seq 1 96); do
  H=$(curl -s -m 5 http://127.0.0.1:8002/health || true)
  case "$H" in
    *'"ready":true'*) echo "LUX READY after ~$((i * 5))s: $H"; exit 0 ;;
    *'"status":"error"'*) echo "LOAD ERROR: $H"; docker logs lux9b 2>&1 | tr '\r' '\n' | grep -i 'error\|failed' | tail -3; exit 1 ;;
  esac
  sleep 5
done
echo "TIMEOUT waiting for lux9b; last health: $H"
docker logs lux9b 2>&1 | tr '\r' '\n' | grep -v 'GET /health' | tail -5
exit 1
