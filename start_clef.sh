#!/usr/bin/env bash
# Start the dockerized Clef-Flash inference API via Docker Compose.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
ENV_FILE=${ENV_FILE:-"$SCRIPT_DIR/clef.env"}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PORT=${PORT:-8001}
HF_HOME_HOST=${HF_HOME_HOST:-./data/huggingface}
BUILD=${BUILD:-0}

mkdir -p "$HF_HOME_HOST"

COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$SCRIPT_DIR/compose.yaml")

if [[ "$BUILD" == "1" ]]; then
  echo "Building image (BUILD=1)..."
  "${COMPOSE[@]}" build
fi

# compose builds automatically if the image does not exist yet.
"${COMPOSE[@]}" up -d --remove-orphans

echo "Started 'clef-flash'."
echo "Endpoint: http://127.0.0.1:$PORT"
echo "Health:   curl -s http://127.0.0.1:$PORT/health"
echo "Logs:     docker compose --env-file $ENV_FILE -f $SCRIPT_DIR/compose.yaml logs -f"
echo "Stop:     $SCRIPT_DIR/stop_clef.sh"
