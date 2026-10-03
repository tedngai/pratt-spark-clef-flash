#!/usr/bin/env bash
# Wait for the clef-flash container to become ready, then report entrypoint/env/health.
set -u
cd /home/tngai/data/clef-flash

echo "entrypoint: $(docker inspect clef-flash --format '{{json .Config.Entrypoint}}')"
docker inspect clef-flash --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(CLEF_SERVER_FILE|BATCH_MAX|BATCH_WAIT_MS|QUEUE_MAX)=' || echo "(no engine env found)"

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/ready 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY after ~$((i * 5))s"
    curl -s http://127.0.0.1:8001/health
    echo
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT waiting for /ready"
docker compose --env-file clef.env -f compose.yaml logs --tail 40 clef-flash
exit 1
